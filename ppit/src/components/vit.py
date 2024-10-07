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
class vit(ViT):
    def __init__(self, *args, **kwargs):
        # Call the parent class (ViT) constructor with the provided arguments
        super(vit, self).__init__(*args, **kwargs)
        self.model = ViT(*args, **kwargs)
        # # self.model = None
        # # self.config = None
        # # self.config_filename, self.config, self.model_dir = load_config(configuration_path, folder="model")
        self.model_dir = "../../../model/model_ViT_test/"
        return

    # def build(self, **kwargs):
    #     try:
    #         print(self.config.image_size)
    #         self.model = ViT(image_size=self.config.image_size,
    #                         patch_size=self.config.patch_size,
    #                         num_classes=self.config.num_classes,
    #                         dim=self.config.dim,
    #                         depth=self.config.depth,
    #                         heads=self.config.heads,
    #                         mlp_dim=self.config.mlp_dim,
    #                         dropout=self.config.dropout,
    #                         emb_dropout=self.config.emb_dropout)
    #         logging.info(f"Finished building ViT")
    #         return
    #     except Exception as e:
    #         raise CustomException(e, sys)

    def loss_fn(self):
        return torch.nn.MSELoss()

    def optimizer(self):
        return torch.optim.Adam(self.model.parameters())

    def train_one_epoch(self,
                        epoch,
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
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report TODO
            epoch_loss += loss

        logging.info(
                    f"Epoch : {epoch + 1} "
                    f"- loss : {epoch_loss / (i + 1):.4f} "
                )

        return epoch_loss / (i + 1)
    def fit(self, X_train, X_test, y_train, y_test):
        # DataLoader
        from torch.utils.data import DataLoader, TensorDataset

        train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=True)


        logging.info(f"Start fitting process")


        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summarywriter = SummaryWriter(f"Model saved at {self.model_dir}/runs/trainer_{timestamp}")
        epoch_number = 0

        EPOCHS = 200


        # loss function
        loss_fn = self.loss_fn()
        # optimizer
        optimizer = self.optimizer()

        for epoch in tqdm(range(EPOCHS)):
            logging.info('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch,
                                            train_dataloader,
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
                    voutputs = self.model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            summarywriter.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            summarywriter.flush()

            # # Track best performance, and save the model's state
            # if avg_vloss < best_vloss:
            #     best_vloss = avg_vloss
            #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            #     torch.save(model.state_dict(), model_path)

        return
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(vit, self).forward(x)  # Call Parent's forward method
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":

    v = vit(
        image_size=24,
        patch_size=4,
        num_classes=48,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,

    )

    import numpy as np
    X_train = torch.tensor(np.random.rand(256, 3, 16, 24), dtype=torch.float32)
    X_test = torch.tensor(np.random.rand(64, 3, 16, 24), dtype=torch.float32)
    y_train = torch.tensor(np.random.rand(256, 48), dtype=torch.float32)
    y_test = torch.tensor(np.random.rand(64, 48), dtype=torch.float32)

    v.fit(X_train, X_test, y_train, y_test)
    output = v(X_train)
    from matplotlib import pyplot as plt
    plt.hist(output.detach().numpy(), label="prediction")
    plt.legend()
    plt.show()
    plt.hist(y_train.detach().numpy(), label="y_test")
    plt.legend()
    plt.show()