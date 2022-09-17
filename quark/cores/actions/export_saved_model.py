# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Provides the `ExportSavedModel` action and associated helper classes."""

import re

from typing import Callable, Optional

import tensorflow as tf


def _id_key(filename):
    _, id_num = filename.rsplit('-', maxsplit=1)
    return int(id_num)


def _find_managed_files(base_name):
    managed_file_regex = re.compile(rf'{re.escape(base_name)}-\d+$')
    filenames = tf.io.gfile.glob(f'{base_name}-*')
    filenames = filter(managed_file_regex.match, filenames)
    return sorted(filenames, key=_id_key)


class _CounterIdFn:
    def __init__(self, base_name: str):
        managed_files = _find_managed_files(base_name)
        self.value = _id_key(managed_files[-1]) + 1 if managed_files else 0

    def __call__(self):
        output = self.value
        self.value += 1
        return output


class ExportFileManager:
    def __init__(self,
                 base_name: str,
                 max_to_keep: int = 5,
                 next_id_fn: Optional[Callable[[], int]] = None):
        self._base_name = base_name
        self._max_to_keep = max_to_keep
        self._next_id_fn = next_id_fn or _CounterIdFn(base_name)

    @property
    def managed_files(self):
        return _find_managed_files(self._base_name)

    def clean_up(self):
        if self._max_to_keep < 0:
            return

        for filename in self.managed_files[:-self._max_to_keep]:
            tf.io.gfile.rmtree(filename)

    def next_name(self) -> str:
        return f'{self._base_name}-{self._next_id_fn()}'


class ExportSavedModel:
    def __init__(self,
                 model: tf.Module,
                 file_manager: ExportFileManager,
                 signatures,
                 options: Optional[tf.saved_model.SaveOptions] = None):
        self.model = model
        self.file_manager = file_manager
        self.signatures = signatures
        self.options = options

    def __call__(self, _):
        export_dir = self.file_manager.next_name()
        tf.saved_model.save(self.model, export_dir, self.signatures,
                            self.options)
        self.file_manager.clean_up()
