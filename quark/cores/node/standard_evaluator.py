# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
from quark.cores import factory
from quark.cores.node.dag import AbstractEvaluator
from quark.cores.node.standard_trainer import product_LabelOptions
from quark.cores import StandardTrainerOptions
from quark.cores import LabelOptions
import tensorflow as tf
import abc
from absl import logging
from typing import List


@factory.register_task_cls('StandardEvaluator')
class StandardEvaluator(AbstractEvaluator, metaclass=abc.ABCMeta):
    """ returns a model fn"""

    def __init__(self, params: StandardTrainerOptions):
        self._model_dag_name = params.model_dag_name
        self._checkpoint_dir = params.checkpoint_dir
        self._max_to_keep = params.max_to_keep
        self._checkpoint_interval = params.checkpoint_interval
        self._interval_pre_step = params.interval_pre_step
        self.global_step = params.global_step
        self._label_config = params.label_config  # 不同的 label_obj 需要定制化loss, metric

    def handler(self, ds: List[tf.data.Dataset]) -> dict:
        # init model info
        self._init_model()
        self._init_loss()
        self._init_metric()
        self._init_restore_checkpoint()

        test_ds = ds[0]
        # eval
        return self._eval_step(test_ds)

    def _init_model(self):
        conf = factory.get_model_cls(self._model_dag_name)
        self._model = conf.model

    def _init_loss(self):
        self._loss = {}
        for conf in self._label_config:
            if conf.loss_method is None:
                continue
            cls = factory.get_loss_cls(conf.loss_method.method)
            self._loss[f"{conf.name}"] = cls(
                conf.loss_method).handler()

    def _init_metric(self):
        self._metric_loss = tf.keras.metrics.Mean(name="loss")
        self._metrics = {}
        for conf in self._label_config:
            if conf.metric_method is None:
                continue
            cls = factory.get_metrics_cls(conf.metric_method.method)
            self._metrics[f"{conf.name}"] = cls(
                conf.metric_method).handler()

    def _init_restore_checkpoint(self):
        checkpoint = tf.train.Checkpoint(model=self._model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self._checkpoint_dir,
            max_to_keep=self._max_to_keep)
        status = checkpoint.restore(self.manager.latest_checkpoint)
        logging.info(f"mdoel reload status: {status}")

    def _eval_step(self, ds):
        # start
        def _loop_begin():
            self._metric_loss.reset_states()
            for metric, metric_fn in self._metrics.items():
                metric_fn.reset_states()

        # end
        def _loop_end():
            result = {
                self._metric_loss.name: self._metric_loss.result().numpy()
            }
            for metric, metric_fn in self._metrics.items():
                result[metric + "_" +
                       metric_fn.name] = metric_fn.result().numpy()
            return result

        # loop
        def _loop(inputs):
            X, y = inputs
            logits = self._model(X, training=True)
            scaled_loss = 0
            for lc in self._label_config:
                name = lc.name
                weight = lc.weight
                if y.get(name) is None:
                    continue
                if logits.get(name) is None:
                    continue
                scaled_loss += weight * tf.reduce_sum(self._loss[name](
                    y[name], logits[name]))
                self._metrics[name].update_state(y[name], logits[name])
                self._metric_loss.update_state(scaled_loss)

        # training
        step = 0
        try:
            inputs = iter(ds)
            _loop_begin()
            while True:
                step += 1
                _loop(next(inputs))
                if step % self._interval_pre_step == 0:
                    result = _loop_end()
                    logging.info(f"\neval -{step} : {result}\n")
        except (StopIteration, tf.errors.OutOfRangeError):
            print(
                f"The eval dataset iterator is exhausted after {step} steps.")
        result = _loop_end()
        logging.info(f"\neval over-{step} : {result}\n")
        return result


def product_StandardEvaluatorOptions(config):
    _conf_name = config.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _model_dag_name = config.get("model_dag_name", None)
    if _model_dag_name is None:
        raise ValueError("`model_dag_name` can be specified.")

    _label_config = config.get("label_config")
    if _label_config is None:
        raise ValueError("`label_config` can be specified.")

    _interval_pre_step = config.get("interval_pre_step", 1000)
    _checkpoint_dir = config.get("checkpoint_dir", "./")
    _checkpoint_interval = config.get("checkpoint_interval", True)
    _max_to_keep = config.get("max_to_keep", 2)

    _label_config_list = []
    for lc in _label_config:
        if lc.get("reuse", False):
            continue
        product_LabelOptions(lc)

    for lc in _label_config:
        _lc_conf_name = lc.get("conf_name")
        _label_config_list.append(factory.get_train_cls(_lc_conf_name))

    @factory.register_task_cls(_conf_name)
    class StandardEvaluatorOptionsNode(StandardTrainerOptions):
        conf_name: str = _conf_name
        model_dag_name: str = _model_dag_name
        interval_pre_step: int = _interval_pre_step
        checkpoint_dir: str = _checkpoint_dir
        max_to_keep: str = _max_to_keep
        checkpoint_interval: bool = _checkpoint_interval
        label_config: List[LabelOptions] = _label_config_list

    logging.info(f"\nregister_train_cls: {_conf_name} is success\n")
