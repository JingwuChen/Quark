# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from tensorflow.python.keras import metrics
import tensorflow as tf


class Metrics():

    def __init__(self, method, **args):
        self.method = method
        self.threshold = args.get("threshold", 0.5)
        self.topk = args.get("topk", 5)
        self.name = args.get("name", "M1")
        self.dtype = args.get("dtype", tf.float32)

        if self.method == "ACC":
            self.m = metrics.Accuracy(name=self.name)

        if self.method == "BACC":
            self.m = metrics.BinaryAccuracy(name=self.name,
                                            threshold=self.threshold)

        if self.method == "CACC":
            self.m = metrics.CategoricalAccuracy(name=self.name)

        if self.method == "SCACC":
            self.m = metrics.SparseCategoricalAccuracy(name=self.name)

        if self.method == "TCACC":
            self.m = metrics.TopKCategoricalAccuracy(name=self.name,
                                                     k=self.topk)

        if self.method == "STCACC":
            self.m = metrics.SparseTopKCategoricalAccuracy(name=self.name,
                                                           k=self.topk)

        if self.method == "PRE":
            self.m = metrics.Precision(name=self.name)

        if self.method == "AUC":
            self.m = metrics.AUC(name=self.name)

        if self.method == "MEAN":
            self.m = metrics.Mean(name=self.name, dtype=self.dtype)

    def get_metrics(self):
        return self.m
