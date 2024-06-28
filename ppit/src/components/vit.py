import random

from ppit.src.exception import CustomException
from ppit.src.logger import logging
import sys
# from vit_pytorch import ViT
from ppit.src.utils import load_config, none_or_str, has_nan, writer, reader
import torch
class vit():
    def __init__(self, configuration_path):
        super().__init__()

        self.model = None
        self.config = None
        self.config_filename, self.config, self.model_dir = load_config(configuration_path, folder="model")

    def build(self, **kwargs):
        try:
            print(self.config.image_size)
            self.model = ViT(image_size=self.config.image_size,
                            patch_size=self.config.patch_size,
                            num_classes=self.config.num_classes,
                            dim=self.config.dim,
                            depth=self.config.depth,
                            heads=self.config.heads,
                            mlp_dim=self.config.mlp_dim,
                            dropout=self.config.dropout,
                            emb_dropout=self.config.emb_dropout)
            logging.info(f"Finished building ViT")
            return
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X_train, X_test, y_train, y_test):

        logging.info(f"Start fitting process")
        logging.info(f"Start turning array to tensor with shape {X_train.shape}")
        X_train = torch.tensor(X_train[:2], dtype=torch.float32)
        logging.info(f"Finish turning array to tensor with shape {X_train.shape}")
        result = self.model(X_train)
        print(result.shape)
        logging.info(f"Finished model prediction")
        return self.model(X_train)

if __name__ == "__main__":
    from vit_pytorch import ViT

    v = ViT(
        image_size=24,
        patch_size=4,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )

    import numpy as np
    img = np.random.rand(100, 3, 16, 24)
    img = torch.tensor(img, dtype=torch.float32)
    preds = v(img)
    print(preds)