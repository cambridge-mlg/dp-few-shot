# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet architecture as used in BiT."""

import tensorflow as tf
from . import normalization
import typing
from keras.utils.layer_utils import count_params
import keras


def add_name_prefix(name, prefix=None):
  return prefix + "/" + name if prefix else name


class ReLU(tf.keras.layers.ReLU):

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)


class PaddingFromKernelSize(tf.keras.layers.Layer):
  """Layer that adds padding to an image taking into a given kernel size."""

  def __init__(self, kernel_size, **kwargs):
    super(PaddingFromKernelSize, self).__init__(**kwargs)
    self._kernel_size = kernel_size
    pad_total = kernel_size - 1
    self._pad_beg = pad_total // 2
    self._pad_end = pad_total - self._pad_beg

  def compute_output_shape(self, input_shape):
    batch_size, height, width, channels = tf.TensorShape(input_shape).as_list()
    if height is not None:
      height = height + self._pad_beg + self._pad_end
    if width is not None:
      width = width + self._pad_beg + self._pad_end
    return tf.TensorShape((batch_size, height, width, channels))

  def call(self, x):
    padding = [
        [0, 0],
        [self._pad_beg, self._pad_end],
        [self._pad_beg, self._pad_end],
        [0, 0]]
    return tf.pad(x, padding)

  def get_config(self):
      config = super().get_config()
      config.update(
          {
              "kernel_size": self._kernel_size,
          }
      )
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)


class StandardizedConv2D(tf.keras.layers.Conv2D):
  """Implements the abs/1903.10520 technique (see go/dune-gn).

  You can simply replace any Conv2D with this one to use re-parametrized
  convolution operation in which the kernels are standardized before conv.

  Note that it does not come with extra learnable scale/bias parameters,
  as those used in "Weight normalization" (abs/1602.07868). This does not
  matter if combined with BN/GN/..., but it would matter if the convolution
  was used standalone.

  Author: Lucas Beyer
  """

  def build(self, input_shape):
    super(StandardizedConv2D, self).build(input_shape)
    # Wrap a standardization around the conv OP.
    default_conv_op = self.convolution_op

    def standardized_conv_op(inputs, kernel):
      # Kernel has shape HWIO, normalize over HWI
      mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
      # Author code uses std + 1e-5
      return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

    self._convolution_op = standardized_conv_op
    self.built = True


class BottleneckV2Unit(tf.keras.layers.Layer):
  """Implements a standard ResNet's unit (version 2).
  """

  def __init__(self, num_filters, stride=1, learnable_params='all', **kwargs):
    """Initializer.

    Args:
      num_filters: number of filters in the bottleneck.
      stride: specifies block's stride.
      **kwargs: other tf.keras.layers.Layer keyword arguments.
    """
    super(BottleneckV2Unit, self).__init__(**kwargs)
    self._num_filters = num_filters
    self._stride = stride
    self.learnable_params = learnable_params

    self._proj = None

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()

    self._unit_a = tf.keras.Sequential([
        normalization.GroupNormalization(name="group_norm", trainable=self.learnable_params == 'all'),
        ReLU(),
    ], name="a")
    self._unit_a_conv = StandardizedConv2D(
        filters=self._num_filters,
        kernel_size=1,
        use_bias=False,
        padding="VALID",
        trainable=self.learnable_params == 'all',
        name="a/standardized_conv2d")

    self._unit_b = tf.keras.Sequential([
        normalization.GroupNormalization(name="group_norm", trainable=self.learnable_params == 'all'),
        ReLU(),
        PaddingFromKernelSize(kernel_size=3),
        StandardizedConv2D(
            filters=self._num_filters,
            kernel_size=3,
            strides=self._stride,
            use_bias=False,
            padding="VALID",
            trainable=self.learnable_params == 'all',
            name="standardized_conv2d")
    ], name="b")

    self._unit_c = tf.keras.Sequential([
        normalization.GroupNormalization(name="group_norm", trainable=self.learnable_params != 'head'),
        ReLU(),
        StandardizedConv2D(
            filters=4 * self._num_filters,
            kernel_size=1,
            use_bias=False,
            padding="VALID",
            trainable=self.learnable_params == 'all',
            name="standardized_conv2d")
    ], name="c")

    if (self._stride > 1) or (4 * self._num_filters != input_shape[-1]):
      self._proj = StandardizedConv2D(
          filters=4 * self._num_filters,
          kernel_size=1,
          strides=self._stride,
          use_bias=False,
          padding="VALID",
          trainable=self.learnable_params == 'all',
          name="a/proj/standardized_conv2d")

    self.built = True

  def compute_output_shape(self, input_shape):
    current_shape = self._unit_a.compute_output_shape(input_shape)
    current_shape = self._unit_a_conv.compute_output_shape(current_shape)
    current_shape = self._unit_b.compute_output_shape(current_shape)
    current_shape = self._unit_c.compute_output_shape(current_shape)
    return current_shape

  def call(self, x):
    x_shortcut = x
    # Unit "a".
    x = self._unit_a(x)
    if self._proj is not None:
      x_shortcut = self._proj(x)
    x = self._unit_a_conv(x)
    # Unit "b".
    x = self._unit_b(x)
    # Unit "c".
    x = self._unit_c(x)

    return x + x_shortcut

  def get_config(self):
      config = super().get_config()
      config.update(
          {
              "num_filters": self._num_filters,
              "stride": self._stride,
              "learnable_params": self.learnable_params,
          }
      )
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)


class ResnetV2(tf.keras.Model):
  """Generic ResnetV2 architecture, as used in the BiT paper."""

  def __init__(self,
               num_units=(3, 4, 6, 3),
               num_outputs=1000,
               filters_factor=4,
               strides=(1, 2, 2, 2),
               learnable_params='all',
               **kwargs):
    super(ResnetV2, self).__init__(**kwargs)

    self.learnable_params = learnable_params

    num_blocks = len(num_units)
    num_filters = tuple(16 * filters_factor * 2**b for b in range(num_blocks))

    self.data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255.0),
            tf.keras.layers.Resizing(height=256, width=256),
            tf.keras.layers.RandomCrop(height=224, width=224),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.Normalization(axis=-1, mean=[0.5, 0.5, 0.5], variance=[0.5 ** 2, 0.5 ** 2, 0.5 ** 2]),
        ],
        name="data_augmentation",
    )

    self._root = self._create_root_block(num_filters=num_filters[0])
    self._blocks = []
    for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
      n = "block{}".format(b)
      self._blocks.append(
          self._create_block(num_units=u, num_filters=f, stride=s, name=n))
    self._pre_head = [
        normalization.GroupNormalization(name="group_norm", trainable=self.learnable_params != 'head'),
        ReLU(),
        tf.keras.layers.GlobalAveragePooling2D()
    ]
    self._head = None
    if num_outputs:
      self._head = tf.keras.layers.Dense(
          units=num_outputs,
          use_bias=True,
          kernel_initializer="zeros",
          trainable=True,
          name="head/dense")

  def _create_root_block(self,
                         num_filters,
                         conv_size=7,
                         conv_stride=2,
                         pool_size=3,
                         pool_stride=2):
    layers = [
        PaddingFromKernelSize(conv_size),
        StandardizedConv2D(
            filters=num_filters,
            kernel_size=conv_size,
            strides=conv_stride,
            trainable=self.learnable_params == 'all',
            use_bias=False,
            name="standardized_conv2d"),
        PaddingFromKernelSize(pool_size),
        tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_stride, padding="valid")
    ]
    return tf.keras.Sequential(layers, name="root_block")

  def _create_block(self, num_units, num_filters, stride, name):
    layers = []
    for i in range(1, num_units + 1):
      layers.append(
          BottleneckV2Unit(
              num_filters=num_filters,
              stride=(stride if i == 1 else 1),
              name="unit%02d" % i,
              learnable_params=self.learnable_params))

    return tf.keras.Sequential(layers, name=name)

  def compute_output_shape(self, input_shape):
    current_shape = self._root.compute_output_shape(input_shape)
    for block in self._blocks:
      current_shape = block.compute_output_shape(current_shape)
    for layer in self._pre_head:
      current_shape = layer.compute_output_shape(current_shape)
    if self._head is not None:
      batch_size, features = current_shape.as_list()
      current_shape = (batch_size, 1, 1, features)
      current_shape = self._head.compute_output_shape(current_shape).as_list()
      current_shape = (current_shape[0], current_shape[3])
    return tf.TensorShape(current_shape)

  def call(self, x):
    x = self.data_augmentation(x)
    x = self._root(x)
    for block in self._blocks:
      x = block(x)
    for layer in self._pre_head:
      x = layer(x)
    if self._head is not None:
      x = self._head(x)
    return x


KNOWN_MODELS = {
    f'{bit}-R{l}x{w}': f'gs://bit_models/{bit}-R{l}x{w}.h5'
    for bit in ['BiT-S', 'BiT-M']
    for l, w in [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
}

NUM_UNITS = {
    k: (3, 4, 6, 3) if 'R50' in k else
       (3, 4, 23, 3) if 'R101' in k else
       (3, 8, 36, 3)
    for k in KNOWN_MODELS
}


def create_block(num_units, num_filters, stride, name, learnable_params):
    layers = []
    for i in range(1, num_units + 1):
        layers.append(
            BottleneckV2Unit(
                num_filters=num_filters,
                stride=(stride if i == 1 else 1),
                name="unit%02d" % i,
                learnable_params=learnable_params))

    return tf.keras.Sequential(layers, name=name)

def resnetv2(
        input_shape=(256, 256),
        num_units=(3, 4, 6, 3),
        strides=(1, 2, 2, 2),
        num_outputs=100,
        filters_factor=4,
        name="resnet",
        learnable_params='all',
        dtype=tf.float32):

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255.0),
            tf.keras.layers.Resizing(height=256, width=256),
            tf.keras.layers.RandomCrop(height=224, width=224),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.Normalization(axis=-1, mean=[0.5, 0.5, 0.5], variance=[0.5 ** 2, 0.5 ** 2, 0.5 ** 2]),
        ],
        name="data_augmentation",
    )

    num_blocks = len(num_units)
    num_filters = tuple(16 * filters_factor * 2 ** b for b in range(num_blocks))

    inputs = keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))
    augmented = data_augmentation(inputs)

    conv_size = 7
    conv_stride = 2
    pool_size = 3
    pool_stride = 2

    y = tf.keras.Sequential([
        PaddingFromKernelSize(conv_size),
        StandardizedConv2D(
            filters=num_filters[0],
            kernel_size=conv_size,
            strides=conv_stride,
            trainable=learnable_params == 'all',
            use_bias=False,
            name="standardized_conv2d"),
        PaddingFromKernelSize(pool_size),
        tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_stride, padding="valid")], name="root_block")(augmented)

    for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
        n = "block{}".format(b)
        y = create_block(num_units=u, num_filters=f, stride=s, name=n, learnable_params=learnable_params)(y)

    y = normalization.GroupNormalization(name="group_norm", trainable=learnable_params != 'head')(y)
    y = ReLU()(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dense(
            units=num_outputs,
            use_bias=True,
            kernel_initializer="zeros",
            trainable=True,
            name="head/dense")(y)
    return tf.keras.models.Model(inputs=inputs, outputs=y, name=name)


def flair_resnet50(
    input_shape: typing.Tuple[int, int, int],
    num_classes: int,
    pretrained: bool,
    adaptation='all',
    seed: int = 0
):

    model = resnetv2(input_shape=input_shape, num_outputs=num_classes, learnable_params=adaptation)

    return model
