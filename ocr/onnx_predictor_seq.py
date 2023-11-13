import os

import numpy as np
import torch

import onnxruntime
from vietocr.model.vocab import Vocab

from common_utils.constants import ONNX_PROVIDERS
from ocr import Cfg
from ocr.predictor import CustomPredictor


class OnnxSeqPredictor(CustomPredictor):
    def __init__(self, model_dir, device):
        config = Cfg.load_config_from_file(os.path.join(model_dir, 'ocr_transformer', self.config_file),
                                           base_config=os.path.join(model_dir, 'ocr_transformer', self.base_file))
        config['cnn']['pretrained'] = False
        config['predictor']['beamsearch'] = False

        config['device'] = device
        config['weights'] = None

        super().__init__(config)
        providers = ONNX_PROVIDERS[device]

        cnn_path = os.path.join(model_dir, 'ocr_seq2seq', 'cnn.onnx')
        self.cnn_session = onnxruntime.InferenceSession(cnn_path, providers=providers)

        encoder_path = os.path.join(model_dir, 'ocr_seq2seq', 'encoder.onnx')
        self.encoder_session = onnxruntime.InferenceSession(encoder_path, providers=providers)

        decoder_path = os.path.join(model_dir, 'ocr_seq2seq', 'decoder.onnx')
        self.decoder_session = onnxruntime.InferenceSession(decoder_path, providers=providers)

        self.device = config['device']
        self.vocab = Vocab(self.config['vocab'])

    def custom_translate(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        cnn_input = {self.cnn_session.get_inputs()[0].name: np.array(img)}
        src = self.cnn_session.run(None, cnn_input)

        encoder_input = {self.encoder_session.get_inputs()[0].name: src[0]}
        encoder_outputs, hidden = self.encoder_session.run(None, encoder_input)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(
                np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = translated_sentence

            decoder_input = {self.decoder_session.get_inputs()[0].name: tgt_inp[-1],
                             self.decoder_session.get_inputs()[1].name: hidden,
                             self.decoder_session.get_inputs()[2].name: encoder_outputs}

            output, hidden, _ = self.decoder_session.run(None, decoder_input)

            output = np.expand_dims(output, axis=1)
            output = torch.Tensor(output)

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
        if (char_probs > 0).sum(-1) == 0:
            char_probs = [0]
        else:
            char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

        return translated_sentence, char_probs
