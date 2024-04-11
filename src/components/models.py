import tensorflow as tf
import numpy as np
from utils import reader, ROOT, AttrDict
import os
import sys
from src.exception import CustomException
from src.logger import logging

class Conv3dBlock:
    def __init__(self,
                 filters=32,
                 kernel_size=(3, 3, 4),
                 strides=(3, 2, 2),
                 padding='same',
                 activation='tanh',
                 dropout=2,
                 name='conv3d_block'
                 ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name
        self.activation = activation
        self.dropout = dropout

    def __call__(self, x):
        x = tf.keras.layers.Conv3D(self.filters,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name=self.name + "_conv3d")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,
                                               name=self.name + "_batchNorm")(x)
        x = tf.keras.layers.Activation(self.activation,
                                       name=self.name + "_activation")(x)
        x = tf.keras.layers.Dropout(self.dropout,
                                    name=self.name + "_dropout")(x)
        logging.info(f"Built Conv3D block {self.name}")
        return x


class Conv2dBlock:
    def __init__(self,
                 filters=32,
                 kernel_size=(2, 2),
                 strides=(1, 1),
                 padding='same',
                 activation='tanh',
                 dropout=2,
                 name='conv3d_block'
                 ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name
        self.activation = activation
        self.dropout = dropout

    def __call__(self, x):
        x = tf.keras.layers.Conv2D(self.filters,
                                   self.kernel_size,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name=self.name + "_conv2d")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,
                                               name=self.name + "_batchNorm")(x)
        x = tf.keras.layers.Activation(self.activation,
                                       name=self.name + "_activation")(x)
        x = tf.keras.layers.Dropout(self.dropout,
                                    name=self.name + "_dropout")(x)

        logging.info(f"Built Conv2D block {self.name}")
        return x


class FlattenBlock:
    def __init__(self,
                 reshape=(128, 12, 3),
                 activation='tanh',
                 dropout=2,
                 name='conv3d_block'):
        self.reshape = reshape
        self.name = name
        self.activation = activation
        self.dropout = dropout

    def __call__(self, x):
        x = tf.keras.layers.Flatten()(x)
        if self.reshape is not None:
            x = tf.keras.layers.Dense(np.prod(self.reshape), name=self.name + "_resize")(x)
        x = tf.keras.layers.Activation(self.activation, name=self.name + "_activation")(x)
        x = tf.keras.layers.Dropout(self.dropout, name=self.name + "_dropout")(x)
        if self.reshape is not None:
            x = tf.keras.layers.Reshape((*self.reshape[1:], self.reshape[0]))(x)

        logging.info(f"Built flatten block {self.name}")
        return x


class csv_logger():

    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        return

    def __call__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.filename)
        logging.info(f"Add callback CSVLogger")
        return tf.keras.callbacks.CSVLogger(log_path)

class save_each_epoch():
    def __init__(self,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=True,
                 mode='auto',
                 period=1
                 ):
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period

    def __call__(self, log_dir):
        self.log_path = os.path.join(log_dir, 'savepoint{epoch:02d}-{val_loss:.2f}')
        logging.info(f"Add callback save each epoch")
        return tf.keras.callbacks.ModelCheckpoint(
                                                    self.log_path,
                                                    monitor=self.monitor,
                                                    verbose=self.verbose,
                                                    save_best_only=self.save_best_only,
                                                    save_weights_only=self.save_weights_only,
                                                    mode=self.mode,
                                                    period=self.period
                                                )


class early_stopping():
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=3,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

    def __call__(self, log_dir):
        logging.info(f"Add callback early stopping")
        return tf.keras.callbacks.EarlyStopping(
                                        monitor=self.monitor,
                                        min_delta=self.min_delta,
                                        patience=self.patience,
                                        verbose=self.verbose,
                                        mode=self.mode,
                                        baseline=self.baseline,
                                        restore_best_weights=self.restore_best_weights,
                                    )
class CNN(tf.keras.Model):
    def __init__(self, configuration_path):
        super().__init__()
        self.model = None
        self.config = None
        self.config_filename = os.path.basename(configuration_path).split(".")[0]
        self.load_config(os.path.join(ROOT,*os.path.split(configuration_path)))
        return

    def load_config(self, configuration_path):
        self.config = AttrDict.from_nested_dicts(reader(configuration_path))

    def build(self, **kwargs):
        try:
            InputLayer = tf.keras.layers.Input(shape=self.config.train.input.shape)
            x = InputLayer
            x = tf.expand_dims(x, axis=-1)

            for block in self.config.model:
                for model, params in block.items():
                    model = eval(model)
                x = model(**params)(x)

            x = tf.keras.layers.Conv2D(1, (2, 2), strides=1, padding='same')(x)
            x = tf.keras.activations.tanh(x) * np.pi

            OutputLayer = x

            self.model = tf.keras.Model(InputLayer, OutputLayer)
            self.model.summary(print_fn=logging.info)

            return
        except Exception as e:
            raise CustomException(e, sys)

    def callbacks(self):
        try:
            log_dir = os.path.join(ROOT, "model", self.config_filename, "log", "training_logs.csv")
            os.makedirs(log_dir, exist_ok=True)
            callback_list = []
            for callback, params in self.config.train.callbacks.items():
                callback = eval(callback)
                callback_list.append(callback(**params)(log_dir))
            return callback_list
        except Exception as e:
            raise CustomException(e, sys)

    def optimizer(self):
        try:

        except Exception as e:
            raise CustomException(e, sys)
    def fit(self):
        try:
            self.model
        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    model = CNN("config/model_params_example.yml")
    model.callbacks()