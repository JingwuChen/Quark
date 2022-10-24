# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import tensor_shape
import tensorflow as tf


class MLP(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 is_batch_norm=False,
                 is_dropout=0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MLP, self).__init__(activity_regularizer=activity_regularizer,
                                  **kwargs)

        self.units = [units] if not isinstance(units, list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims = [last_dim] + self.units

        self.kernels = []
        self.biases = []
        self.bns = []
        for i in range(len(dims) - 1):
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i], dims[i + 1]],
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint,
                                dtype=self.dtype,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[
                                        dims[i + 1],
                                    ],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True))
            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built = True

    def call(self, inputs, is_train=False):
        _input = inputs
        for i in range(len(self.units)):
            _input = tf.linalg.matmul(a=_input, b=self.kernels[i])
            if self.use_bias:
                _input = tf.nn.bias_add(value=_input, bias=self.biases[i])
            # BN
            if self.is_batch_norm:
                _input = self.bns[i](_input)
            # ACT
            if self.activation is not None:
                _input = self.activation(_input)
            # DROP
            if is_train and self.is_dropout > 0:
                _input = tf.keras.layers.Dropout(self.is_dropout)(_input)

        return _input

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            'units':
            self.units,
            'activation':
            tf.keras.activations.serialize(self.activation),
            'use_bias':
            self.use_bias,
            'is_batch_norm':
            self.is_batch_norm,
            'is_dropout':
            self.is_dropout,
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
            tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint)
        })
        return config
