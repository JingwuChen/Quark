# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
from quark.cores import factory
from quark.cores import dir_is_exist_create
from quark.cores.node.dag import AbstractSaveModel
from quark.cores import StandardSaveModelOptions
import tensorflow as tf
import abc
from absl import logging
from typing import List


@factory.register_task_cls('StandardSaveModel')
class StandardSaveModel(AbstractSaveModel, metaclass=abc.ABCMeta):
    """ returns a model fn"""

    def __init__(self, params: StandardSaveModelOptions):
        self._model_dag_name = params.model_dag_name
        self._checkpoint_dir = params.checkpoint_dir
        self._signatures = params.signatures
        self._base_dir = params.base_dir
        self._options = params.options
        self._max_to_keep = params.max_to_keep
        self._next_id = params.next_id
        if self._next_id is None:
            managed_files = self._managed_files()
            self._next_id = self._id_key(
                managed_files[-1]) + 1 if managed_files else 0
        dir_is_exist_create(self._base_dir)

    def handler(self, isOver: List[bool]):
        self._init_model()
        self._init_restore_checkpoint()
        _export_dir = self._next_name()
        tf.saved_model.save(self._model, _export_dir, self._signatures,
                            self._options)
        self._clean_up()
        logging.info(f"\nsave model dir: {_export_dir}\n")

    def _id_key(self, filename):
        _, id_num = filename.rsplit('-', maxsplit=1)
        return int(id_num)

    def _managed_files(self):
        # managed_file_regex = re.compile(
        #     rf'{re.escape({self._base_dir}/{self._model_dag_name})}-\d+$')
        filenames = tf.io.gfile.glob(
            f'{self._base_dir}/{self._model_dag_name}-*')
        # filenames = filter(managed_file_regex.match, filenames)
        return sorted(filenames, key=self._id_key)

    def _clean_up(self):
        if self._max_to_keep < 0:
            return

        for filename in self._managed_files()[:-self._max_to_keep]:
            tf.io.gfile.rmtree(filename)
            logging.info(f"\nclean model dir: {filename}\n")

    def _next_name(self) -> str:
        return f'{self._base_dir}/{self._model_dag_name}-{self._next_id}'

    def _init_model(self):
        conf = factory.get_model_cls(self._model_dag_name)
        self._model = conf.model

    def _init_restore_checkpoint(self):
        checkpoint = tf.train.Checkpoint(model=self._model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self._checkpoint_dir,
            max_to_keep=self._max_to_keep)
        status = checkpoint.restore(self.manager.latest_checkpoint)
        logging.info(f"mdoel reload status: {status}")


def product_StandardSaveModel(config):
    _conf_name = config.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _model_dag_name = config.get("model_dag_name", None)
    if _model_dag_name is None:
        raise ValueError("`model_dag_name` can be specified.")

    _checkpoint_dir = config.get("checkpoint_dir", "./")
    _max_to_keep = config.get("max_to_keep", 5)
    _base_dir = config.get("base_dir", "./savemodel")

    @factory.register_task_cls(_conf_name)
    class StandardSaveModelOptionsNode(StandardSaveModelOptions):
        conf_name: str = _conf_name
        model_dag_name: str = _model_dag_name
        checkpoint_dir: str = _checkpoint_dir
        max_to_keep: str = _max_to_keep
        base_dir: str = _base_dir

    logging.info(f"\nregister_savemodel_cls: {_conf_name} is success\n")
