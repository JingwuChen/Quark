# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Provides AbstractTrainer/Evaluator base classes, defining train/eval APIs."""


import abc
from typing import Dict, Optional, Union
import numpy as np
import tensorflow as tf

Output = Dict[str, Union[tf.Tensor, float, np.number, np.ndarray, 'Output']]  # pytype: disable=not-supported-yet


class AbstractTrainer(tf.Module, metaclass=abc.ABCMeta):
    """An abstract class defining the API required for training."""

    @abc.abstractmethod
    def train(self, num_steps: tf.Tensor) -> Optional[Output]:
        """Implements `num_steps` steps of training."""
        pass


class AbstractEvaluator(tf.Module, metaclass=abc.ABCMeta):
    """An abstract class defining the API required for evaluation."""

    @abc.abstractmethod
    def evaluate(self, num_steps: tf.Tensor) -> Optional[Output]:
        """Implements `num_steps` steps of evaluation."""
        pass
