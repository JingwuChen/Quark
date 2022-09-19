# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A global DAG to access data config."""
# 整个数据图里，我们包含两个大的步骤
# 0. 原始数据
# 1. 数据导入
# 2. 数据处理
# 3. 训练数据
# --
# 0. 训练数据
# 1. 设置环境
# 2. 设置runner
# 3. 产出结果

import abc
from typing import Optional, Union
import tensorflow as tf

Output = Union[tf.data.Dataset, list[Optional[float], Optional[int]]]  # pytype: disable=not-supported-yet
"""
这里注意, 每实现一个inputer 请将其注册到data_factory,这样就可以共享使用
使用方法为：
@data_factory.register_config_factory('TFRecodFileWith[business name]')
class TFRcordFileInputer(AbstractInputer):
    ...
    ...
"""


class AbstractInputer(metaclass=abc.ABCMeta):
    """An abstract class defining the API required for Inputer."""

    @abc.abstractmethod
    def read(self) -> Optional[Output]:
        """Implements `dtype` of Inputer."""
        pass


class AbstractProcessor(metaclass=abc.ABCMeta):
    """An abstract class defining the API required for Processor."""

    @abc.abstractmethod
    def handler(self, actions: list[str]) -> Optional[Output]:
        """Implements `num_steps` steps of Processor."""
        pass


"""
这里注意, 每实现一个inputer 请将其注册到data_factory,这样就可以共享使用
使用方法为：
@task_factory.register_config_factory('DSSMRecallWith[business name]')
class DSSMTrainer(AbstractTrainer):
    ...
    ...

"""


class AbstractEnvironment(metaclass=abc.ABCMeta):
    """An abstract class defining the API required for Env."""

    @abc.abstractmethod
    def initialize(self):
        pass


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


class AbstractSaveModel(metaclass=abc.ABCMeta):
    """An abstract class defining the API required for SaveModel."""

    @abc.abstractmethod
    def save(self, dtype: str) -> Optional[Output]:
        """Implements `dtype` dtype of save."""
        pass
