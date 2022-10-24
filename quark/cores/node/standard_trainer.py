# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
from quark.cores import factory
from quark.cores.node.dag import AbstractTrainer
from quark.cores import StandardTrainerOptions
from quark.cores import LabelOptions, OptimizerOptions, LosserOptions, MetricOptions
import tensorflow as tf
import abc
from absl import logging
from typing import Union, List, Tuple


@factory.register_task_cls('StandardTrainer')
class StandardTrainer(AbstractTrainer, metaclass=abc.ABCMeta):
    """ returns a model fn"""

    def __init__(self, params: StandardTrainerOptions):
        self._mode = params.mode
        self._model_dag_name = params.model_dag_name
        self.epochs = params.epochs
        self._checkpoint_dir = params.checkpoint_dir
        self._max_to_keep = params.max_to_keep
        self._checkpoint_interval = params.checkpoint_interval
        self._interval_pre_step = params.interval_pre_step
        self._global_step = params.global_step
        self._optimizer_config = params.optimizer_config
        self._label_config = params.label_config  # 不同的 label_obj 需要定制化loss, metric

    def handler(self, ds: List[tf.data.Dataset]):
        # init model info
        self._init_model()
        self._init_loss()
        self._init_optimizer()
        self._init_metric()
        self._init_save_checkpoint()
        train_ds = ds[0]
        test_ds = None
        if len(ds) >= 2:
            test_ds = ds[1]
        # train
        if train_ds is None:
            raise ValueError("dataset is None")
        if self._mode == "train":
            for e in range(self.epochs):
                logging.info(f"\n=====epoch: {e}=====\n")
                self._train_step(train_ds)
                self.manager.save(checkpoint_number=e, check_interval=True)

        # train_and_eval
        if self._mode == "train_and_eval" and test_ds is not None:
            for e in range(self.epochs):
                logging.info(f"\n=====epoch: {e} start=====\n")
                self._train_step(train_ds)
                logging.info(f"\n=====train epoch: {e} end=====")
                self._eval_step(test_ds)
                logging.info(f"\n=====eval epoch: {e} end=====")
                self.manager.save(checkpoint_number=e, check_interval=True)

        return True

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

    def _init_optimizer(self):
        cls = factory.get_optimizer_cls(self._optimizer_config.method)
        self._optimizer = cls(self._optimizer_config).handler()

    def _init_metric(self):
        self._metric_loss = tf.keras.metrics.Mean(name="loss")
        self._metrics = {}
        for conf in self._label_config:
            if conf.metric_method is None:
                continue
            cls = factory.get_metrics_cls(conf.metric_method.method)
            self._metrics[f"{conf.name}"] = cls(
                conf.metric_method).handler()

    def _init_save_checkpoint(self):
        checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,
                                         model=self._model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self._checkpoint_dir,
            max_to_keep=self._max_to_keep,
            checkpoint_interval=self._checkpoint_interval,
            step_counter=self._optimizer.iterations)

    def _train_step(self, ds):
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
                result[metric + "/" +
                       metric_fn.name] = metric_fn.result().numpy()
            return result

        # loop
        def _loop(inputs):
            with tf.GradientTape() as tape:
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
                gradients = tape.gradient(scaled_loss,
                                          self._model.trainable_variables)

                self._optimizer.apply_gradients(
                    list(zip(gradients, self._model.trainable_variables)))

        # train
        step = 0
        try:
            inputs = iter(ds)
            _loop_begin()
            while True:
                step += 1
                _loop(next(inputs))
                if step % self._interval_pre_step == 0:
                    result = _loop_end()
                    logging.info(f"\ntrain -{step} : {result}\n")
        except (StopIteration, tf.errors.OutOfRangeError):
            print(
                f"The train dataset iterator is exhausted after {step} steps.")
        result = _loop_end()
        logging.info(f"\ntrain over-{step} : {result}\n")
        return result

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
                result[metric + "/" +
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

        # eval
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


def product_StandardTrainerOptions(config):
    _conf_name = config.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _mode = config.get("mode", None)
    if _mode is None:
        raise ValueError("`_mode` can be specified.")

    _model_dag_name = config.get("model_dag_name", None)
    if _model_dag_name is None:
        raise ValueError("`model_dag_name` can be specified.")

    _interval_pre_step = config.get("interval_pre_step", 1000)
    _checkpoint_dir = config.get("checkpoint_dir", "./")
    _checkpoint_interval = config.get("checkpoint_interval", True)
    _max_to_keep = config.get("max_to_keep", 2)

    _optimizer_config = config.get("optimizer_config", None)
    if _optimizer_config is None:
        raise ValueError("`optimizer_config` can be specified.")
    if not _optimizer_config.get("reuse", False):
        product_OptimizerOptions(_optimizer_config)
    _optimizer_cls = factory.get_optimizer_cls(_optimizer_config["conf_name"])

    _label_config = config.get("label_config")
    if _label_config is None:
        raise ValueError("`label_config` can be specified.")
    _label_config_list = []
    for lc in _label_config:
        if lc.get("reuse", False):
            continue
        product_LabelOptions(lc)

    for lc in _label_config:
        _lc_conf_name = lc.get("conf_name")
        _label_config_list.append(factory.get_train_cls(_lc_conf_name))

    # if not _model_config.get("reuse", False):
    #     product_ModelOptions(_model_config)
    # _model_cls = factory.get_model_cls(_model_config["conf_name"])

    @factory.register_task_cls(_conf_name)
    class StandardTrainerOptionsNode(StandardTrainerOptions):
        conf_name: str = _conf_name
        mode: str = _mode
        model_dag_name: str = _model_dag_name
        interval_pre_step: int = _interval_pre_step
        checkpoint_dir: str = _checkpoint_dir
        max_to_keep: str = _max_to_keep
        checkpoint_interval: bool = _checkpoint_interval
        optimizer_config: OptimizerOptions = _optimizer_cls
        label_config: List[LabelOptions] = _label_config_list

    logging.info(f"\nregister_train_cls: {_conf_name} is success\n")


def product_LosserOptions(params):
    _conf_name = params.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _method = params.get("method", None)
    if _method is None:
        raise ValueError("`method` can be specified.")

    _from_logits = params.get("from_logits", False)
    _reduction = params.get("reduction", tf.keras.losses.Reduction.NONE)
    _gamma = params.get("gamma", 2.0)
    _label_smoothing = params.get("label_smoothing", 0.0)
    _delta = params.get("delta", 1.0)

    @factory.register_loss_cls(_conf_name)
    class LosserOptionsNode(LosserOptions):
        conf_name: str = _conf_name
        method: str = _method
        from_logits: bool = _from_logits
        reduction = _reduction
        gamma: float = _gamma
        label_smoothing: float = _label_smoothing
        delta: float = _delta

    logging.info(f"\nregister_Losser_cls: {_conf_name} is success\n")


def product_MetricOptions(params):
    _conf_name = params.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _method = params.get("method", None)
    if _method is None:
        raise ValueError("`method` can be specified.")

    _from_logits = params.get("from_logits", False)
    _thresholds = params.get("thresholds", None)
    _label_smoothing = params.get("label_smoothing", 0.0)
    _target_class_ids = params.get("target_class_ids", (0, 1))
    _top_k = params.get("top_k", None)
    _class_id = params.get("class_id", None)
    _rate = params.get("rate", None)
    _num_thresholds = params.get("num_thresholds", 200)
    _k = params.get("k", 5)

    @factory.register_metrics_cls(_conf_name)
    class MetricOptionsNode(MetricOptions):
        conf_name: str = _conf_name
        method: str = _method
        from_logits: bool = _from_logits
        thresholds: float = _thresholds
        label_smoothing: float = _label_smoothing
        target_class_ids: Union[List[int], Tuple[int, ...]] = _target_class_ids
        top_k: int = _top_k
        class_id: int = _class_id
        rate: float = _rate
        num_thresholds: int = _num_thresholds
        k: int = _k

    logging.info(f"\nregister_metrics_cls: {_conf_name} is success\n")


def product_LabelOptions(params):
    _conf_name = params.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _name = params.get("name", None)
    if _name is None:
        raise ValueError("`name` can be specified.")

    _dtype = params.get("dtype", None)

    _weight = params.get("weight", 1)

    _loss_config = params.get("loss_config", None)
    if _loss_config is None:
        raise ValueError("`loss_config` can be specified.")
    if not _loss_config.get("reuse", False):
        product_LosserOptions(_loss_config)
    _loss_method = factory.get_loss_cls(_loss_config["conf_name"])

    _metric_config = params.get("metric_config", None)
    if _metric_config is None:
        raise ValueError("`metric_config` can be specified.")
    if not _metric_config.get("reuse", False):
        product_MetricOptions(_metric_config)
    _metric_method = factory.get_metrics_cls(_metric_config["conf_name"])

    @factory.register_train_cls(_conf_name)
    class LabelOptionsNode(LabelOptions):
        conf_name: str = _conf_name
        name: str = _name
        dtype: str = _dtype
        weight: float = _weight
        loss_method: str = _loss_method
        metric_method: str = _metric_method

    logging.info(f"\nregister_train_cls: {_conf_name} is success\n")


def product_OptimizerOptions(params):
    _conf_name = params.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _method = params.get("method", None)
    if _method is None:
        raise ValueError("`method` can be specified.")

    _learning_rate = params.get("learning_rate", 0.001)
    _beta_1 = params.get("beta_1", 0.9)
    _beta_2 = params.get("beta_2", 0.999)
    _epsilon = params.get("epsilon", 1e-07)
    _amsgrad = params.get("amsgrad", False)
    _momentum = params.get("momentum", 0.0)
    _nesterov = params.get("nesterov", False)
    _rho = params.get("rho", 0.9)
    _centered = params.get("centered", False)
    _learning_rate_power = params.get("learning_rate_power", -0.5)
    _initial_accumulator_value = params.get("initial_accumulator_value", 0.1)
    _l1_regularization_strength = params.get("l1_regularization_strength", 0.0)
    _l2_regularization_strength = params.get("l2_regularization_strength", 0.0)
    _l2_shrinkage_regularization_strength = params.get(
        "l2_shrinkage_regularization_strength", 0.0)
    _beta = params.get("beta", 0.0)
    _weight_decay_rate = params.get("weight_decay_rate", 0.0)
    _eeta = params.get("eeta", 0.001)
    _classic_momentum = params.get("classic_momentum", True)

    @factory.register_optimizer_cls(_conf_name)
    class OptimizerOptionsNode(OptimizerOptions):
        conf_name: str = _conf_name
        method: str = _method
        learning_rate: float = _learning_rate
        beta_1: float = _beta_1
        beta_2: float = _beta_2
        epsilon: float = _epsilon
        amsgrad: bool = _amsgrad
        momentum: float = _momentum
        nesterov: bool = _nesterov
        rho: float = _rho
        centered: bool = _centered
        learning_rate_power: float = _learning_rate_power
        initial_accumulator_value: float = _initial_accumulator_value
        l1_regularization_strength: float = _l1_regularization_strength
        l2_regularization_strength: float = _l2_regularization_strength
        l2_shrinkage_regularization_strength: float = _l2_shrinkage_regularization_strength
        beta: float = _beta
        weight_decay_rate: float = _weight_decay_rate
        eeta: float = _eeta
        classic_momentum: bool = _classic_momentum

    logging.info(f"\nregister_optimizer_cls: {_conf_name} is success\n")
