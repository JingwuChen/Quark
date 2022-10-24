# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from quark.cores import factory


@factory.register_model_cls('product')
class ProductLayer(tf.keras.layers.Layer):

    def __init__(self, conf, **kwargs):
        super(ProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, is_train=False):
        if len(inputs) != 2:
            raise ValueError("`inputs` len must be 2")
        _out = tf.math.reduce_sum(inputs[0] * inputs[1], axis=-1)
        _out = tf.math.sigmoid(_out, name="pred")
        return _out
