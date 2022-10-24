# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Common configuration settings."""

from typing import Optional, Union, List, Text, Tuple
from quark.cores import config as cfg
import tensorflow as tf


class PrintDataSetOptions(cfg.PrintConfig):
    node_name: str = "PrintDataSetOptionsNode"
    read_num: int = 1


class FieldOptions(cfg.DataConfig):
    conf_name: str = "FieldOptions"
    name: str = None
    dtype: int = 0
    len: int = None
    value_dtype: int = None


class StandardInputerOptions(cfg.DataConfig):
    conf_name: str = "StandardInputerOptions"
    mode: str = None
    files: Union[List[str], str] = None
    compression_type: str = None
    buffer_size: int = 100
    num_parallel_reads: int = tf.data.experimental.AUTOTUNE
    feature_config: List[FieldOptions] = None
    textline_split: str = None


class StandardProcessorOptions(cfg.DataConfig):
    conf_name: str = "StandardProcessorOptions"
    global_batch_size: int = 16
    is_training: bool = None
    drop_remainder: bool = True
    shuffle_buffer_size: int = 100
    cache: bool = False
    sharding: bool = True
    plugins: List[str] = None
    context: Optional[tf.distribute.InputContext] = None


class ModelOptions(cfg.ModelConfig):
    conf_name: str = "ModelOptions"
    model: object = None


class OptimizerOptions(cfg.TrainerConfig):
    conf_name: str = "OptimizerOptions"
    method: str = None
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    momentum: float = 0.0
    nesterov: bool = False
    rho: float = 0.9
    centered: bool = False
    learning_rate_power: float = -0.5
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    beta: float = 0.0
    weight_decay_rate: float = 0.0
    eeta: float = 0.001
    classic_momentum: bool = True
    exclude_from_weight_decay: Optional[List[Text]] = None
    exclude_from_layer_adaptation: Optional[List[Text]] = None


class LosserOptions(cfg.TrainerConfig):
    conf_name: str = "LosserOptions"
    method: str = None
    from_logits: bool = False
    reduction = tf.keras.losses.Reduction.NONE
    gamma: float = 2.0
    label_smoothing: float = 0.0
    delta: float = 1.0


class MetricOptions(cfg.TrainerConfig):
    conf_name: str = "MetricOptions"
    method: str = None
    from_logits: bool = False
    thresholds: float = None
    label_smoothing: float = 0.0
    target_class_ids: Union[List[int], Tuple[int, ...]] = (0, 1)
    top_k: int = None
    class_id: int = None
    rate: float = None
    num_thresholds: int = 200
    k: int = 5


class LabelOptions(cfg.TrainerConfig):
    conf_name: str = "LabelOptions"
    name: str = None
    dtype: str = None
    weight: float = 1
    loss_method: LosserOptions = None
    metric_method: MetricOptions = None


class StandardTrainerOptions(cfg.TrainerConfig):
    conf_name: str = "StandardTrainerOptions"
    mode: str = "train"
    model_dag_name: str = None
    epochs: int = 1
    global_step: int = None
    interval_pre_step: int = None
    checkpoint_dir: str = None
    max_to_keep: str = 2
    checkpoint_interval: bool = True
    label_config: List[LabelOptions] = None
    optimizer_config: OptimizerOptions = None


class StandardSaveModelOptions(cfg.TrainerConfig):
    conf_name: str = "StandardSaveModelOptions"
    model_dag_name: str = None
    signatures: str = None
    base_dir: str = None
    options: Optional[tf.saved_model.SaveOptions] = None
    max_to_keep: int = 5
    next_id: int = None
    checkpoint_dir: str = None
