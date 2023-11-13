import os


class Config(object):
    pass


class DevelopmentConfig(Config):
    if os.getenv("MODEL_DIR"):
        MODEL_DIR = os.getenv("MODEL_DIR")
    else:
        MODEL_DIR = "/Users/tieuanhnguyen/PycharmProjects/VND_VAT/vnd-pti-ocr/models/pti_ocr"

    if os.getenv("EXPORT_DIR"):
        EXPORT_DIR = os.getenv("EXPORT_DIR")
    else:
        EXPORT_DIR = "/Users/tieuanhnguyen/PycharmProjects/VND_VAT/data/pti/table_train_vat"


dev_config = DevelopmentConfig
