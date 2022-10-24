# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
from calendar import c

from quark.cores import factory
import tensorflow as tf
from quark.cores import MetricOptions


@factory.register_metrics_cls("auc")
class AUCMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.AUC(name=f"{_conf_name}/{_name}",
                                            thresholds=_thresholds,
                                            from_logits=_from_logits)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("accuracy")
class AccuracyMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        self.metrics = tf.keras.metrics.Accuracy(name=f"{_conf_name}/{_name}")

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("binaryaccuracy")
class BinaryAccuracyMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.BinaryAccuracy(
            name=f"{_conf_name}/{_name}", thresholds=_thresholds)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("binarycrossentropy")
class BinaryCrossentropyMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _label_smoothing = conf.label_smoothing
        self.metrics = tf.keras.metrics.BinaryCrossentropy(
            name=f"{_conf_name}/{_name}",
            from_logits=_from_logits,
            label_smoothing=_label_smoothing)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("binaryiou")
class BinaryIoUMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        _target_class_ids = conf.target_class_ids
        self.metrics = tf.keras.metrics.BinaryIoU(
            name=f"{_conf_name}/{_name}",
            thresholds=_thresholds,
            target_class_ids=_target_class_ids)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("categoricalaccuracy")
class CategoricalAccuracyMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        self.metrics = tf.keras.metrics.CategoricalAccuracy(
            name=f"{_conf_name}/{_name}")

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("categoricalhinge")
class CategoricalHingeMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        self.metrics = tf.keras.metrics.CategoricalHinge(
            name=f"{_conf_name}/{_name}")

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("cosinesimilarity")
class CosineSimilarityMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        self.metrics = tf.keras.metrics.CosineSimilarity(
            name=f"{_conf_name}/{_name}")

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("falsenegatives")
class FalseNegativesMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.FalseNegatives(
            name=f"{_conf_name}/{_name}",
            thresholds=_thresholds,
        )

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("falsepositives")
class FalsePositivesMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.FalsePositives(
            name=f"{_conf_name}/{_name}",
            thresholds=_thresholds,
        )

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("truenegatives")
class TrueNegativesMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.TrueNegatives(
            name=f"{_conf_name}/{_name}",
            thresholds=_thresholds,
        )

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("truepositives")
class TruePositivesMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        self.metrics = tf.keras.metrics.TruePositives(
            name=f"{_conf_name}/{_name}",
            thresholds=_thresholds,
        )

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("precision")
class PrecisionMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        _top_k = conf.top_k
        _class_id = conf.class_id
        self.metrics = tf.keras.metrics.Precision(name=f"{_conf_name}/{_name}",
                                                  thresholds=_thresholds,
                                                  top_k=_top_k,
                                                  class_id=_class_id)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("recall")
class RecallMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _thresholds = conf.thresholds
        _top_k = conf.top_k
        _class_id = conf.class_id
        self.metrics = tf.keras.metrics.Recall(name=f"{_conf_name}/{_name}",
                                               thresholds=_thresholds,
                                               top_k=_top_k,
                                               class_id=_class_id)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("precisionatrecall")
class PrecisionAtRecallMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _num_thresholds = conf.num_thresholds
        _recall = conf.rate
        _class_id = conf.class_id
        self.metrics = tf.keras.metrics.PrecisionAtRecall(
            name=f"{_conf_name}/{_name}",
            num_thresholds=_num_thresholds,
            recall=_recall,
            class_id=_class_id)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("recallatprecision")
class RecallAtPrecisionMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _num_thresholds = conf.num_thresholds
        _precision = conf.rate
        _class_id = conf.class_id
        self.metrics = tf.keras.metrics.RecallAtPrecision(
            name=f"{_conf_name}/{_name}",
            num_thresholds=_num_thresholds,
            precision=_precision,
            class_id=_class_id)

    def handler(self):
        return self.metrics


@factory.register_metrics_cls("topkcategoricalaccuracy")
class TopKCategoricalAccuracyMetrics:

    def __init__(self, conf: MetricOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _k = conf.k

        self.metrics = tf.keras.metrics.TopKCategoricalAccuracy(
            name=f"{_conf_name}/{_name}", k=_k)

    def handler(self):
        return self.metrics
