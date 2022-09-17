# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Customized Relu activation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def relu6(features):
    features = tf.convert_to_tensor(features)
    return tf.nn.relu6(features)
