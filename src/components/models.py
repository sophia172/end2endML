import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from utils import reader, ROOT, AttrDict
import os


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
        if self.reshape is not None:
            x = tf.keras.layers.Dense(np.prod(self.reshape), name=self.name + "_resize")(x)
        x = tf.keras.layers.Activation(self.activation, name=self.name + "_activation")(x)
        x = tf.keras.layers.Dropout(self.dropout, name=self.name + "_dropout")(x)
        if self.reshape is not None:
            x = tf.keras.layers.Reshape((*self.reshape[1:], self.reshape[0]))(x)
        return x

from pathlib import Path
class CNN(tf.keras.Model):
    def __init__(self, configuration_path):
        super().__init__()
        self.config = None
        self.load_config(os.path.join(ROOT,*os.path.split(configuration_path)))
        return

    def load_config(self, configuration_path):
        self.config = AttrDict.from_nested_dicts(reader(configuration_path))

    def build(self, **kwargs):
        InputLayer = tf.keras.layers.Input(shape=self.config.train.input.shape)
        x = InputLayer
        x = tf.expand_dims(x, axis=-1)

        for block in self.config.model:
            for model, params in block.items():
                model = eval(model)
            x = model(**params)(x)
            print(x)

        x = tf.keras.layers.Conv2D(1, (2, 2), strides=1, padding='same')(x)
        x = tf.keras.activations.tanh(x) * np.pi

        OutputLayer = x

        model = tf.keras.Model(InputLayer, OutputLayer)
        return model.summary()


if __name__=="__main__":
    model = CNN("config/model_params_example.yml")
    model.build()