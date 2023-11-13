from ocr.config import Cfg


def get_ocr_predictor(model_dir, model_type, model_name, device, num_torch_threads):
    if model_type == 'seq2seq':
        from ocr.predictor import Seq2SeqCustomPredictor
        return Seq2SeqCustomPredictor(model_dir, model_name, device, num_torch_threads)
    elif model_type == 'transformer_onnx':
        from ocr.onnx_predictor import OnnxCustomPredictor

        return OnnxCustomPredictor(model_dir, device)
    elif model_type == 'seq2seq_onnx':
        from ocr.onnx_predictor_seq import OnnxSeqPredictor

        return OnnxSeqPredictor(model_dir, device)

    from ocr.predictor import TransformerCustomPredictor

    return TransformerCustomPredictor(model_dir, device, num_torch_threads)
