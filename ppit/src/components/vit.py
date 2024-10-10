import random

from ppit.src.exception import CustomException
from ppit.src.logger import logging
import sys
from vit_pytorch import ViT
from ppit.src.utils import load_config, none_or_str, has_nan, writer, reader
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Function for early stopping
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
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class vit():
    def __init__(self, configuration_path):
        # Call the parent class (ViT) constructor with the provided arguments
        super(vit, self).__init__()
        # # self.model = None
        # # self.config = None
        # # self.config_filename, self.config, self.model_dir = load_config(configuration_path, folder="model")
        # self.model_dir = "../../../model/model_ViT_test/"
        # self.epoch = epoch
        # self.batch = batch
        # self.model = ViT(*args, **kwargs)
        self.model = None
        self.config = None
        self.config_filename, self.config, self.model_dir = load_config(configuration_path, folder="model")
        return

    def build(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "mps" if torch.backends.mps.is_available()
                                        else "cpu")

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
            # self.model = ViT(**self.config.model).to(self.device)
            logging.info(f"Finished building ViT")
            return
        except Exception as e:
            raise CustomException(e, sys)

    def loss_fn(self):
        return torch.nn.MSELoss()

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters())

    def train_one_epoch(self,
                        train_dataloader,
                        optimizer=None,
                        loss_fn=None):

        epoch_loss = 0
        epoch_accuracy = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs.to(self.device))

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report TODO
            epoch_loss += loss

        logging.info(
                    f"- loss : {epoch_loss / (i + 1):.4f} "
                )

        return epoch_loss / (i + 1)
    def fit(self, X_train, X_test, y_train, y_test):
        # DataLoader

        train_dataloader = DataLoader(TensorDataset(
                                            torch.from_numpy(X_train.astype(np.float32)),
                                                    torch.from_numpy(y_train.astype(np.float32))
                                                    ),
                                        batch_size=self.config.train.batch_size,
                                        shuffle=True)
        test_dataloader = DataLoader(TensorDataset(
                                            torch.from_numpy(X_test.astype(np.float32)),
                                                    torch.from_numpy(y_test.astype(np.float32))
                                                    ),
                                        batch_size=self.config.train.batch_size,
                                        shuffle=True)

        logging.info(f"Start fitting process")


        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summarywriter = SummaryWriter(f"Model saved at {self.model_dir}/runs/trainer_{timestamp}")
        epoch_number = 0


        # loss function
        loss_fn = self.loss_fn()
        # optimizer
        optimizer = self.optimizer()

        # Instantiate the EarlyStopping object
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        for epoch in tqdm(range(self.config.train.epochs)):
            logging.info('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(train_dataloader,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            )

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs.to(self.device))
                    vloss = loss_fn(voutputs, vlabels.to(self.device))
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Call early stopping and break if the condition is met
            early_stopping(avg_vloss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break


            # Log the running loss averaged per batch
            # for both training and validation
            summarywriter.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            summarywriter.flush()
        self.save()
        return


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(vit, self).forward(x)  # Call Parent's forward method
        x = torch.sigmoid(x)
        return x

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        self.model.eval()  # Set model to evaluation mode
        predictions = []

        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        dataloader = DataLoader(dataset, batch_size=self.config.train.batch_size, shuffle=False)

        with torch.no_grad():  # Disable gradient calculation
            for inputs in dataloader:
                inputs = inputs[0].to(self.device)  # Move inputs to GPU
                outputs = self.model(inputs)  # Perform forward pass
                predictions.append(outputs.cpu())  # Move results back to CPU

        # Concatenate all predictions into a single tensor
        return torch.cat(predictions).detach().numpy()

    def save(self):
        try:
            torch.save(self.model, os.path.join(self.model_dir, "model.pth"))
            logging.info(f"Model saved at {self.model_dir}")
        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":

    v = vit(
        image_size=(14,24),
        patch_size=2,
        num_classes=48,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        epoch=10,
        batch=256,
        configuration_path="../../../config/"

    )

    import numpy as np
    X_train = np.random.rand(256, 3, 14, 24)
    X_test = np.random.rand(64, 3, 14, 24)
    y_train = np.random.rand(256, 48)
    y_test = np.random.rand(64, 48)

    X_val = np.random.rand(64, 3, 14, 24)
    v.fit(X_train, X_test, y_train, y_test)
    output = v.predict(X_val)
    from matplotlib import pyplot as plt
    plt.hist(output, label="prediction")
    plt.legend()
    plt.show()
    plt.hist(y_train, label="y_test")
    plt.legend()
    plt.show()