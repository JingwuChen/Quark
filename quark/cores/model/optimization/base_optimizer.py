# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

from quark.cores import factory
import tensorflow as tf
from quark.cores import OptimizerOptions
from quark.cores.model.optimization.lars_optimizer import LARS


@factory.register_optimizer_cls("adam")
class AdamOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _beta_1 = conf.beta_1
        _beta_2 = conf.beta_2
        _epsilon = conf.epsilon
        _amsgrad = conf.amsgrad
        self.optim = tf.keras.optimizers.Adam(name=f"{_conf_name}/{_name}",
                                              learning_rate=_learning_rate,
                                              beta_1=_beta_1,
                                              beta_2=_beta_2,
                                              epsilon=_epsilon,
                                              amsgrad=_amsgrad)

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("sgd")
class SgdOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _momentum = conf.momentum
        _nesterov = conf.nesterov
        self.optim = tf.keras.optimizers.SGD(name=f"{_conf_name}/{_name}",
                                             learning_rate=_learning_rate,
                                             momentum=_momentum,
                                             nesterov=_nesterov)

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("rms")
class RmsOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _momentum = conf.momentum
        _rho = conf.rho
        _epsilon = conf.epsilon
        _centered = conf.centered
        self.optim = tf.keras.optimizers.RMSprop(name=f"{_conf_name}/{_name}",
                                                 learning_rate=_learning_rate,
                                                 momentum=_momentum,
                                                 rho=_rho,
                                                 epsilon=_epsilon,
                                                 centered=_centered)

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("nadam")
class NadamOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _beta_1 = conf.beta_1
        _beta_2 = conf.beta_2
        _epsilon = conf.epsilon
        self.optim = tf.keras.optimizers.Nadam(name=f"{_conf_name}/{_name}",
                                               learning_rate=_learning_rate,
                                               beta_1=_beta_1,
                                               beta_2=_beta_2,
                                               epsilon=_epsilon)

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("ftrl")
class FtrlOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _learning_rate_power = conf.learning_rate_power
        _initial_accumulator_value = conf.initial_accumulator_value
        _l1_regularization_strength = conf.l1_regularization_strength
        _l2_regularization_strength = conf.l2_regularization_strength
        _l2_shrinkage_reg_strength = conf.l2_shrinkage_regularization_strength
        self.optim = tf.keras.optimizers.Ftrl(
            name=f"{_conf_name}/{_name}",
            learning_rate=_learning_rate,
            learning_rate_power=_learning_rate_power,
            initial_accumulator_value=_initial_accumulator_value,
            l1_regularization_strength=_l1_regularization_strength,
            l2_regularization_strength=_l2_regularization_strength,
            l2_shrinkage_regularization_strength=_l2_shrinkage_reg_strength,
        )

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("adagrad")
class AdagradOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _initial_accumulator_value = conf.initial_accumulator_value
        _epsilon = conf.epsilon
        self.optim = tf.keras.optimizers.Adagrad(
            name=f"{_conf_name}/{_name}",
            learning_rate=_learning_rate,
            initial_accumulator_value=_initial_accumulator_value,
            epsilon=_epsilon)

    def handler(self):
        return self.optim


@factory.register_optimizer_cls("lars")
class LarsOptimizer:

    def __init__(self, conf: OptimizerOptions):
        _conf_name = conf.conf_name
        _name = conf.method
        _learning_rate = conf.learning_rate
        _momentum = conf.momentum
        _weight_decay_rate = conf.weight_decay_rate
        _eeta = conf.eeta
        _nesterov = conf.nesterov
        _classic_momentum = conf.classic_momentum
        _exclude_from_layer_adaptation = conf.exclude_from_layer_adaptation
        _exclude_from_weight_decay = conf.exclude_from_weight_decay
        self.optim = LARS(
            name=f"{_conf_name}/{_name}",
            learning_rate=_learning_rate,
            momentum=_momentum,
            weight_decay_rate=_weight_decay_rate,
            eeta=_eeta,
            nesterov=_nesterov,
            classic_momentum=_classic_momentum,
            exclude_from_weight_decay=_exclude_from_weight_decay,
            exclude_from_layer_adaptation=_exclude_from_layer_adaptation,
        )

    def handler(self):
        return self.optim
