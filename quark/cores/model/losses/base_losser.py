# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from quark.cores import factory
import tensorflow as tf
from quark.cores import LosserOptions


@factory.register_loss_cls("bce")
class BinaryCrossentropyLosser:
    """
    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

        y_true (true label): This is either 0 or 1.
        y_pred (predicted value): This is the model's prediction,
        i.e, a single floating-point value which either represents a logit,
        (i.e, value in [-inf, inf] when from_logits=True) or
        a probability (i.e, value in [0., 1.] when from_logits=False).

        Recommended Usage: (set from_logits=True)
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _reduction = conf.reduction
        self.losser = tf.keras.losses.BinaryCrossentropy(
            name=f"{_conf_name}/{_name}",
            from_logits=_from_logits,
            reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("bfce")
class BinaryFocalCrossentropyLosser:
    """
    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

        y_true (true label): This is either 0 or 1.
        y_pred (predicted value): This is the model's prediction,
        i.e, a single floating-point value which either represents a logit,
        (i.e, value in [-inf, inf] when from_logits=True) or
        a probability (i.e, value in [0., 1.] when from_logits=False).

        focal_factor = (1 - output) ** gamma for class 1
        focal_factor = output ** gamma for class 0
        where gamma is a focusing parameter.
        When gamma=0, this function is equivalent to the binary crossentropy loss.
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _reduction = conf.reduction
        _gamma = conf.gamma
        _label_smoothing = conf.label_smoothing
        self.losser = tf.keras.losses.BinaryFocalCrossentropy(
            name=f"{_conf_name}/{_name}",
            gamma=_gamma,
            from_logits=_from_logits,
            label_smoothing=_label_smoothing,
            reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("cce")
class CategoricalCrossentropyLosser:
    """
    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a one_hot representation.
    If you want to provide labels as integers,
    please use SparseCategoricalCrossentropy loss.
    There should be # classes floating point values per feature.

    In the snippet below, there is # classes floating pointing values per
    example. The shape of both y_pred and y_true
    are [batch_size, num_classes].
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _reduction = conf.reduction
        self.losser = tf.keras.losses.CategoricalCrossentropy(
            name=f"{_conf_name}/{_name}",
            from_logits=_from_logits,
            reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("ch")
class CategoricalHingeLosser:
    """
    loss = maximum(neg - pos + 1, 0)
    where neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> h = tf.keras.losses.CategoricalHinge()
    >>> h(y_true, y_pred).numpy()
    1.4

    >>> # Calling with 'sample_weight'.
    >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
    0.6
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.CategoricalHinge(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("cs")
class CosineSimilarityLosser:
    """
    Note that it is a number between -1 and 1.
    When it is a negative number between -1 and 0,
    0 indicates orthogonality and values closer
    to -1 indicate greater similarity.
    The values closer to 1 indicate greater dissimilarity.
    This makes it usable as a loss function in a setting
    where you try to maximize the proximity between
    predictions and targets. If either y_true or y_pred is
    a zero vector, cosine similarity will be 0 regardless of
    the proximity between predictions and targets.

    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.CosineSimilarity(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("hinge")
class HingeLosser:
    """
    loss = maximum(1 - y_true * y_pred, 0)

    y_true values are expected to be -1 or 1. If binary (0 or 1) labels
    are provided we will convert them to -1 or 1.
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> h = tf.keras.losses.Hinge()
    >>> h(y_true, y_pred).numpy()
    1.3

    >>> # Calling with 'sample_weight'.
    >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
    0.55
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.Hinge(name=f"{_conf_name}/{_name}",
                                            reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("huber")
class HuberLosser:
    """
    For each value x in error = y_true - y_pred:
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _delta = conf.delta
        _reduction = conf.reduction
        self.losser = tf.keras.losses.Huber(name=f"{_conf_name}/{_name}",
                                            delta=_delta,
                                            reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("kldivergence")
class KLDivergenceLosser:
    """
    loss = y_true * log(y_true / y_pred)
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.KLDivergence(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("logcosh")
class LogCoshLosser:
    """
    logcosh = log((exp(x) + exp(-x))/2), where x is the error y_pred - y_true.
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.LogCosh(name=f"{_conf_name}/{_name}",
                                              reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("mape")
class MeanAbsolutePercentageErrorLosser:
    """
    loss = 100 * abs((y_true - y_pred) / y_true)
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.MeanAbsolutePercentageError(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("mse")
class MeanSquaredErrorLosser:
    """
    loss = square(y_true - y_pred)
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.MeanSquaredError(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("msle")
class MeanSquaredLogarithmicErrorLosser:
    """
    loss = square(log(y_true + 1.) - log(y_pred + 1.))
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.MeanSquaredLogarithmicError(
            name=f"{_conf_name}/{_name}", reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("poisson")
class PoissonLosser:
    """
    loss = y_pred - y_true * log(y_pred)
    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _reduction = conf.reduction
        self.losser = tf.keras.losses.Poisson(name=f"{_conf_name}/{_name}",
                                              reduction=_reduction)

    def handler(self):
        return self.losser


@factory.register_loss_cls("scce")
class SparseCategoricalCrossentropyLosser:
    """
    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided as integers. If you want to
    provide labels using one-hot representation, please use
    CategoricalCrossentropy loss. There should be # classes floating
    point values per feature for y_pred and a single floating point
    value per feature for y_true.

    In the snippet below, there is a single floating point value
    per example for y_true and # classes floating pointing
    values per example for y_pred. The shape of y_true
    is [batch_size] and the shape of y_pred is [batch_size, num_classes].

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
    >>> scce(y_true, y_pred).numpy()

    """

    def __init__(self, conf: LosserOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _from_logits = conf.from_logits
        _reduction = conf.reduction
        self.losser = tf.keras.losses.SparseCategoricalCrossentropy(
            name=f"{_conf_name}/{_name}",
            from_logits=_from_logits,
            reduction=_reduction)

    def handler(self):
        return self.losser
