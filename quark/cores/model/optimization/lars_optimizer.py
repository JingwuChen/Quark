# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Layer-wise adaptive rate scaling optimizer."""
import re
from typing import Text, List, Optional

import tensorflow as tf

# pylint: disable=protected-access


class LARS(tf.keras.optimizers.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay_rate: float = 0.0,
                 eeta: float = 0.001,
                 nesterov: bool = False,
                 classic_momentum: bool = True,
                 exclude_from_weight_decay: Optional[List[Text]] = None,
                 exclude_from_layer_adaptation: Optional[List[Text]] = None,
                 name: Text = "LARS",
                 **kwargs):
        super(LARS, self).__init__(name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", self._initial_decay)
        self.momentum = momentum
        self.weight_decay_rate = weight_decay_rate
        self.eeta = eeta
        self.nesterov = nesterov
        self.classic_momentum = classic_momentum
        self.exclude_from_weight_decay = exclude_from_weight_decay
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "momentum")

    def _resource_apply_dense(self, grad, param, apply_state=None):
        if grad is None or param is None:
            return tf.no_op()

        var_device, var_dtype = param.device, param.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        learning_rate = coefficients["lr_t"]

        param_name = param.name

        v = self.get_slot(param, "momentum")

        if self._use_weight_decay(param_name):
            grad += self.weight_decay_rate * param

        if self.classic_momentum:
            trust_ratio = 1.0
            if self._do_layer_adaptation(param_name):
                w_norm = tf.norm(param, ord=2)
                g_norm = tf.norm(grad, ord=2)
                trust_ratio = tf.where(
                    tf.greater(w_norm, 0),
                    tf.where(tf.greater(g_norm, 0),
                             (self.eeta * w_norm / g_norm), 1.0), 1.0)
            scaled_lr = learning_rate * trust_ratio

            next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
            if self.nesterov:
                update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
            else:
                update = next_v
            next_param = param - update
        else:
            next_v = tf.multiply(self.momentum, v) + grad
            if self.nesterov:
                update = tf.multiply(self.momentum, next_v) + grad
            else:
                update = next_v

            trust_ratio = 1.0
            if self._do_layer_adaptation(param_name):
                w_norm = tf.norm(param, ord=2)
                v_norm = tf.norm(update, ord=2)
                trust_ratio = tf.where(
                    tf.greater(w_norm, 0),
                    tf.where(tf.greater(v_norm, 0),
                             (self.eeta * w_norm / v_norm), 1.0), 1.0)
            scaled_lr = trust_ratio * learning_rate
            next_param = param - scaled_lr * update

        return tf.group(*[
            param.assign(next_param, use_locking=False),
            v.assign(next_v, use_locking=False)
        ])

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError(
            "Applying sparse gradients is not implemented.")

    def _use_weight_decay(self, param_name):
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def get_config(self):
        config = super(LARS, self).get_config()
        config.update({
            "learning_rate":
            self._serialize_hyperparameter("learning_rate"),
            "decay":
            self._serialize_hyperparameter("decay"),
            "momentum":
            self.momentum,
            "classic_momentum":
            self.classic_momentum,
            "weight_decay_rate":
            self.weight_decay_rate,
            "eeta":
            self.eeta,
            "nesterov":
            self.nesterov,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
