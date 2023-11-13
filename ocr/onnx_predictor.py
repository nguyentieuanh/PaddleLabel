import os

import numpy as np
from torchvision import transforms
import onnxruntime
from vietocr.model.vocab import Vocab

from common_utils.constants import ONNX_PROVIDERS
from ocr import Cfg
from ocr.predictor import CustomPredictor


class OnnxCustomPredictor(CustomPredictor):
    def __init__(self, model_dir, device):
        config = Cfg.load_config_from_file(os.path.join(model_dir, 'ocr', self.config_file),
                                           base_config=os.path.join(model_dir, 'ocr', self.base_file))
        config['cnn']['pretrained'] = False
        config['predictor']['beamsearch'] = False

        config['device'] = device
        config['weights'] = None

        super().__init__(config)
        providers = ONNX_PROVIDERS[device]

        encoder_path = os.path.join(model_dir, 'ocr', 'transformer_encoder.onnx')
        self.encoder_session = onnxruntime.InferenceSession(encoder_path, providers=providers)

        decoder_path = os.path.join(model_dir, 'ocr', 'transformer_decoder.onnx')
        self.decoder_session = onnxruntime.InferenceSession(decoder_path, providers=providers)
        self.convert_tensor = transforms.ToTensor()

        self.device = config['device']
        self.vocab = Vocab(self.config['vocab'])

    def custom_translate(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        # include cnn and encoder of vietocr
        onnx_inp = {self.encoder_session.get_inputs()[0].name: np.array(img, dtype=np.float32)}
        onnx_out = np.squeeze(self.encoder_session.run(None, onnx_inp))
        onnx_out = self.convert_tensor(onnx_out)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(
                np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = translated_sentence

            # decoder of vietocr
            onnx_inp = {self.decoder_session.get_inputs()[0].name: np.array(tgt_inp),
                        self.decoder_session.get_inputs()[1].name: np.array(onnx_out, dtype=np.float32)}
            onnx_values, onnx_indices = self.decoder_session.run(None, onnx_inp)

            onnx_indices = onnx_indices[:, -1, 0]
            onnx_indices = onnx_indices.tolist()

            onnx_values = onnx_values[:, -1, 0]
            onnx_values = onnx_values.tolist()
            char_probs.append(onnx_values)

            translated_sentence.append(onnx_indices)
            max_length += 1

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

        return translated_sentence, char_probs
