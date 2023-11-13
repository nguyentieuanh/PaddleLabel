import os

import numpy as np
from PIL import Image
from PIL import Image as im
import torch
from torch.nn.functional import softmax
from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import process_input, translate_beam_search

from ocr import Cfg

from pipeline.base_obj import BaseDetector


# from pipeline.base_obj import BaseDetector

class CustomPredictor(Predictor, BaseDetector):
    config_file = 'config.yml'
    base_file = 'base.yml'

    def __init__(self, config):
        if config['weights'] is None:
            return
        if config['num_torch_threads'] is not None:
            torch.set_num_threads(config['num_torch_threads'])
        self.num_torch_threads = config['num_torch_threads']
        super().__init__(config)
        self.device = config['device']
        self.model.eval()

    def custom_translate(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        with torch.no_grad():
            if self.num_torch_threads is not None:
                torch.set_num_threads(self.num_torch_threads)

            src = self.model.cnn(img)
            memory = self.model.transformer.forward_encoder(src)

            translated_sentence = [[sos_token] * len(img)]
            char_probs = [[1] * len(img)]
            max_length = 0

            while max_length <= max_seq_length and not all(
                    np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
                tgt_inp = torch.LongTensor(translated_sentence).to(self.device)

                output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)
                output = softmax(output, dim=-1)
                output = output.to('cpu')

                values, indices = torch.topk(output, 5)

                indices = indices[:, -1, 0]
                indices = indices.tolist()

                values = values[:, -1, 0]
                values = values.tolist()
                char_probs.append(values)

                translated_sentence.append(indices)
                max_length += 1

                del output

            translated_sentence = np.asarray(translated_sentence).T

            char_probs = np.asarray(char_probs).T
            char_probs = np.multiply(char_probs, translated_sentence > 3)
            char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

        return translated_sentence, char_probs

    def predict(self, img):
        if type(img) is str:
            img = Image.open(img)
        if isinstance(img, np.ndarray):
            img = im.fromarray(img)
        img = process_input(img, self.config['dataset']['image_height'],
                            self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = self.custom_translate(img)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        return s, prob


class Seq2SeqCustomPredictor(CustomPredictor):
    def __init__(self, model_dir, model_name, device, num_torch_threads):
        config = Cfg.load_config_from_file(os.path.join(model_dir, 'ocr_seq2seq', self.config_file),
                                           base_config=os.path.join(model_dir, 'ocr_seq2seq', self.base_file))
        config['cnn']['pretrained'] = False
        config['device'] = device
        config['predictor']['beamsearch'] = False
        config['num_torch_threads'] = num_torch_threads
        config['weights'] = os.path.join(model_dir, 'ocr_seq2seq', f"{model_name}.pth")

        super().__init__(config)


class TransformerCustomPredictor(CustomPredictor):
    def __init__(self, model_dir, device, num_torch_threads):
        config = Cfg.load_config_from_file(os.path.join(model_dir, 'ocr_transformer', self.config_file),
                                           base_config=os.path.join(model_dir, 'ocr_transformer', self.base_file))
        config['cnn']['pretrained'] = False
        config['device'] = device
        config['predictor']['beamsearch'] = False
        config['num_torch_threads'] = num_torch_threads
        config['weights'] = os.path.join(model_dir, 'ocr', 'vgg_transformer_ocr.pth')

        super().__init__(config)


class CaptchaCustomPredictor(CustomPredictor):
    def __init__(self, model_dir, model_name='vgg_seq2seq_captcha', device='cpu', num_torch_threads=2):
        config = Cfg.load_config_from_file(os.path.join(model_dir, 'config.yml'),
                                           os.path.join(model_dir, 'base.yml'))
        config['cnn']['pretrained'] = False
        config['device'] = device
        config['predictor']['beamsearch'] = False
        config['num_torch_threads'] = num_torch_threads
        config['weights'] = os.path.join(model_dir, f'{model_name}.pth')

        super().__init__(config)
