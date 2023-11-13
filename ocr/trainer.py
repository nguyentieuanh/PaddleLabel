import os.path
import sys
import time
from torch.optim.lr_scheduler import OneCycleLR
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.model.trainer import Trainer
from vietocr.tool.translate import build_model
from vietocr.tool.logger import Logger
from torch.optim import AdamW
from vietocr.loader.aug import ImgAugTransform
from vietocr.tool.utils import download_weights

EARLY_STOP_FUNC = {
    'acc_full_seq': (lambda a, b: a > b),
    'acc_per_char': (lambda a, b: a > b),
    'val_loss': (lambda a, b: a < b),
}


class CustomTrainer(Trainer):
    def __init__(self, config, pretrained=True):

        self.config = config
        self.model, self.vocab = build_model(config)

        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']

        if 'checkpoint' not in config['trainer'] and 'export' not in config['trainer']:
            self.export_weights = './best_model'
        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']

        self.metrics = config['trainer']['metrics']
        # self.early_stopping = config['trainer']['early_stopping']

        self.early_stopping_var = config['early_stopping_var']
        if self.early_stopping_var in EARLY_STOP_FUNC:
            self.early_stopping_func = EARLY_STOP_FUNC[self.early_stopping_var]
        else:
            self.early_stopping_func = EARLY_STOP_FUNC['acc_full_seq']

        logger = config['trainer']['log']

        if logger:
            self.logger = Logger(logger)

        if pretrained:
            # Change: load from local
            if config['weights'].startswith('http'):
                weight_file = download_weights(**config['pretrain'], quiet=config['quiet'])
            else:
                weight_file = config['weights']
            self.load_weights(weight_file)

        self.iter = 0

        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        transforms = ImgAugTransform()

        # Change: move folder to inside data
        self.train_gen = self.data_gen(config['dataset']['train_path'],
                                       self.data_root, self.train_annotation, transform=transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen(config['dataset']['val_path'],
                                           self.data_root, self.valid_annotation)

        self.train_losses = []

    def train(self):
        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0

        data_iter = iter(self.train_gen)
        patience_count = 0
        best_var = float("inf") if self.early_stopping_var == 'val_loss' else 0

        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(
                    self.iter,
                    total_loss / self.print_every, self.optimizer.param_groups[0]['lr'],
                    total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                    self.iter, val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                var_dict = {
                    'acc_full_seq': acc_full_seq,
                    'acc_per_char': acc_per_char,
                    'val_loss': val_loss,
                }

                compare_var = var_dict[self.early_stopping_var]
                if self.early_stopping_func(compare_var, best_var):
                    if self.export_weights:
                        print("Save best weight {} {}".format(self.early_stopping_var, compare_var))
                        self.save_weights(self.export_weights)
                    else:
                        print("Save checkpoint checkpoint_{:06d}_{:.3f}_{:.4f}_{:.4f}.pth".format(
                            self.iter, val_loss, acc_full_seq, acc_per_char))
                        self.save_weights(
                            os.path.join(self.checkpoint, "checkpoint_{:06d}_{:.3f}_{:.4f}_{:.4f}.pth".format(
                                self.iter, val_loss, acc_full_seq, acc_per_char)))
                    patience_count = 0
                    best_var = compare_var
                else:
                    patience_count += 1
                    print("Did not improve, patience increase to {}".format(patience_count))

                # Change: add early stop
                if acc_full_seq >= self.config['max_acc'] or acc_per_char >= self.config['max_acc']:
                    mess = "{} Early stop because of accuracy >= {}".format(time.ctime(), self.config['max_acc'])
                    print(mess)
                    self.logger.log(mess)
                    sys.exit()

                if patience_count >= self.config['patience']:
                    mess = "{} Early stop because of patience >= {}".format(time.ctime(), self.config['patience'])
                    print(mess)
                    self.logger.log(mess)
                    sys.exit()
