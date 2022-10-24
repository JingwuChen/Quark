# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from quark.cores import factory


@factory.register_model_cls('input')
class InputsLayer(tf.keras.layers.Layer):

    def __init__(self, conf, **kwargs):
        super(InputsLayer, self).__init__(**kwargs)
        self._name = conf["feature_name"]
        _reuse = conf.get("reuse", False)
        self._value_dtype = conf.get("value_dtype", None)
        self._len = conf.get("len", None)
        if _reuse:
            feat = factory.get_data_cls(self._name)
            self._value_dtype = feat.value_dtype
            self._len = feat.len

    def build(self, input_shape):
        self.built = True

    def call(self, inputs=None, is_train=False):
        if self._value_dtype == 1:
            tmp = tf.keras.Input(shape=(self._len, ),
                                 name=f"{self._name}",
                                 dtype=tf.float64)
            return tmp

        if self._value_dtype == 2:
            tmp = tf.keras.Input(shape=(self._len, ),
                                 name=f"{self._name}",
                                 dtype=tf.string)
            return tmp

        if self._value_dtype == 3:
            tmp = tf.keras.Input(shape=(self._len, ),
                                 name=f"{self._name}",
                                 dtype=tf.int64)
            return tmp
