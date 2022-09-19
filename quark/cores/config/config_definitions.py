# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Common configuration settings."""

import dataclasses
from typing import Optional, Union, List


@dataclasses.dataclass
class FeatuerConfig():
    name: str = None
    dtype: int = 0
    len: int = None
    value_dtype: int = None


@dataclasses.dataclass
class DataConfig():
    # inputer basic config
    tfds_name: str = None
    mode: str = None
    files: Union[List[str], str] = None
    compression_type: str = None
    buffer_size: int = 100
    num_parallel_reads: int = None
    feature_config: List[FeatuerConfig] = None
    textline_split: str = None
    # processor basic config
    global_batch_size: int = 0
    is_training: bool = None
    drop_remainder: bool = True
    shuffle_buffer_size: int = 100
    cache: bool = False
    cycle_length: Optional[int] = None
    block_length: int = 1
    deterministic: Optional[bool] = None
    sharding: bool = True
    enable_tf_data_service: bool = False
    tf_data_service_address: Optional[str] = None
    tf_data_service_job_name: Optional[str] = None
    seed: Optional[int] = None


@dataclasses.dataclass
class RuntimeConfig():
    distribution_strategy: str = "mirrored"
    enable_xla: bool = False
    gpu_thread_mode: Optional[str] = None
    dataset_num_private_threads: Optional[int] = None
    per_gpu_thread_count: int = 0
    tpu: Optional[str] = None
    num_gpus: int = 0
    worker_hosts: Optional[str] = None
    task_index: int = -1
    all_reduce_alg: Optional[str] = None
    num_packs: int = 1
    mixed_precision_dtype: Optional[str] = None
    loss_scale: Optional[Union[str, float]] = None
    run_eagerly: bool = False
    batchnorm_spatial_persistent: bool = False
    tpu_enable_xla_dynamic_padder: Optional[bool] = None
    num_cores_per_replica: int = 1
    default_shard_dim: int = -1

    def model_parallelism(self):
        return dict(num_cores_per_replica=self.num_cores_per_replica,
                    default_shard_dim=self.default_shard_dim)


@dataclasses.dataclass
class TrainerConfig():
    train_tf_while_loop: bool = True
    train_tf_function: bool = True
    eval_tf_function: bool = True
    eval_tf_while_loop: bool = False
    allow_tpu_summary: bool = False
    steps_per_loop: int = 1000
    summary_interval: int = 1000
    checkpoint_interval: int = 1000
    max_to_keep: int = 5
    continuous_eval_timeout: int = 60 * 60
    train_steps: int = 0
    validation_steps: int = -1
    validation_interval: int = 1000
    best_checkpoint_export_subdir: str = ""
    best_checkpoint_eval_metric: str = ""
    best_checkpoint_metric_comp: str = "higher"
    loss_upper_bound: float = 1e6
    recovery_begin_steps: int = 0
    recovery_max_trials: int = 0
    validation_summary_subdir: str = "validation"


@dataclasses.dataclass
class TaskConfig():
    init_checkpoint: str = ""
    model: Optional[str] = None
    train_data: DataConfig = DataConfig()
    validation_data: DataConfig = DataConfig()
    name: Optional[str] = None


@dataclasses.dataclass
class ExperimentConfig():
    """Top-level configuration."""
    task: TaskConfig = TaskConfig()
    trainer: TrainerConfig = TrainerConfig()
    runtime: RuntimeConfig = RuntimeConfig()
