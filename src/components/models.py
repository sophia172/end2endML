import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from utils import reader


@dataclass
class ModelConfig(dict):
    model: list
    train: dict

    def __getattr__(self, item):
        return self.get(item)


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
        x = tf.keras.layers.Dense(np.prod(self.reshape), name=self.name + "_resize")(x)
        x = tf.keras.layers.Activation(self.activation, name=self.name + "_activation")(x)
        x = tf.keras.layers.Dropout(self.dropout, name=self.name + "_dropout")(x)
        if self.reshape is not None:
            x = tf.keras.layers.Reshape((self.reshape[1:], self.reshape[0]))(x)
        return x


class CNN(tf.keras.Model):
    def __init__(self, configuration_path):
        super().__init__()
        self.load_config(configuration_path)
        return

    def load_config(self, configuration_path):
        self.config = ModelConfig(**reader(configuration_path))

    def build(self, **kwargs):
        InputLayer = tf.keras.layers.Input()
        x = InputLayer

        for block in self.config.model:
            model = eval(block)
            x = model(**block)(x)

        x = tf.keras.layers.Conv2D(1, (2, 2), strides=1, padding='same')(x)
        x = tf.keras.activations.tanh(x) * np.pi

        OutputLayer = x

        return InputLayer, OutputLayer
