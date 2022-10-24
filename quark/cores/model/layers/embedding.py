# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from quark.cores import factory


@factory.register_model_cls('embed')
class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, conf, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.feature_size = conf["feature_size"]
        self.embedding_size = conf["embedding_size"]

    def build(self, input_shape):
        # embedding
        self.embed = tf.keras.layers.Embedding(self.feature_size,
                                               self.embedding_size,
                                               embeddings_regularizer="l2")
        self.built = True

    def call(self, inputs, is_train=False):

        return self.embed(inputs)
