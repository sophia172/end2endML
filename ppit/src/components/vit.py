import os
import sys
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT

from ppit.src.exception import CustomException
from ppit.src.logger import logging
from ppit.src.utils import load_config, none_or_str, has_nan, writer, reader


# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: How many epochs to wait after last time validation loss improved.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Vision Transformer Wrapper Class
class ViTTrainer:
    def __init__(self, configuration_path):
        super().__init__()
        self.model = None
        self.config = None
        self.config_filename, self.config, self.model_dir = load_config(configuration_path, folder="model")
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")

    def build(self):
        try:
            self.model = ViT(
                image_size=tuple(self.config.model.image_size),
                patch_size=self.config.model.patch_size,
                num_classes=self.config.model.num_classes,
                dim=self.config.model.dim,
                depth=self.config.model.depth,
                heads=self.config.model.heads,
                mlp_dim=self.config.model.mlp_dim,
                dropout=self.config.model.dropout,
                emb_dropout=self.config.model.emb_dropout,
            ).to(self.device)
            logging.info("Finished building ViT")
        except Exception as e:
            raise CustomException(e, sys)

    def loss_fn(self):
        return torch.nn.MSELoss()

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters())

    def dataloader(self, X, y=None):
        batch_size = 1 if len(X) == 1 else self.config.train.batch_size
        dataset = TensorDataset(
            *(torch.from_numpy(X.astype(np.float32)),) if y is None else
            (torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_one_epoch(self, train_dataloader, optimizer, loss_fn):
        epoch_loss = 0
        self.model.train(True)

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = self.model(inputs.to(self.device))
            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logging.info(f"- Loss: {epoch_loss / (i + 1):.4f}")
        return epoch_loss / (i + 1)

    def fit(self, X_train, X_test, y_train, y_test):
        train_dataloader = self.dataloader(X_train, y_train)
        test_dataloader = self.dataloader(X_test, y_test)
        logging.info("Start fitting process")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summarywriter = SummaryWriter(f"{self.model_dir}/runs/trainer_{timestamp}")
        loss_fn = self.loss_fn()
        optimizer = self.optimizer()
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        for epoch in tqdm(range(self.config.train.epochs)):
            logging.info(f'EPOCH {epoch + 1}:')
            avg_loss = self.train_one_epoch(train_dataloader, optimizer, loss_fn)

            running_vloss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs.to(self.device))
                    vloss = loss_fn(voutputs, vlabels.to(self.device))
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            logging.info(f"Train Loss: {avg_loss:.4f}, Validation Loss: {avg_vloss:.4f}")

            summarywriter.add_scalars(
                'Training vs. Validation Loss',
                {'Training': avg_loss, 'Validation': avg_vloss},
                epoch + 1
            )
            summarywriter.flush()

            if early_stopping(avg_vloss):
                logging.info("Early stopping triggered")
                break

        self.save()

    def save(self):
        try:
            torch.save(self.model, os.path.join(self.model_dir, "model.pth"))
            logging.info(f"Model saved at {self.model_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in self.dataloader(X):
                inputs = inputs[0].to(self.device)
                y_pred = self.model(inputs)
                predictions.append(y_pred.cpu())

        return torch.cat(predictions).detach().numpy()


if __name__ == "__main__":
    v = ViTTrainer(configuration_path="../../../config/model_ViT_example.yml")
    v.build()

    X_train = np.random.rand(256, 3, 14, 24)
    X_test = np.random.rand(64, 3, 14, 24)
    y_train = np.random.rand(256, 48)
    y_test = np.random.rand(64, 48)
    X_val = np.random.rand(64, 3, 14, 24)

    v.fit(X_train, X_test, y_train, y_test)
    output = v.predict(X_val)

    plt.hist(output, label="Prediction")
    plt.legend()
    plt.show()
    plt.hist(y_train, label="y_train")
    plt.legend()
    plt.show()
