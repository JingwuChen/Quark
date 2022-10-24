# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
from quark.cores import factory
from quark.cores.node.dag import AbstractModel
import abc
from absl import logging
from quark.cores import ModelOptions
import tensorflow as tf


@factory.register_task_cls('StandardModel')
class StandardModel(AbstractModel, metaclass=abc.ABCMeta):

    def __init__(self, params):
        self.model_dag = params

    def instantiate(self):
        self.node_2cls_dict = {
            "start": {
                "cls": None,
                "name": "start",
                "input": None,
                "output": [],
                "dag_is_input": False,
                "dag_is_output": False,
            }
        }
        for node in self.model_dag:
            _name = node["node_name"]
            _dtype = node["node_dtype"]
            _input = node["node_input"].split(",")
            _dag_is_input = node.get("dag_is_input", False)
            _dag_is_output = node.get("dag_is_output", False)
            _config = node["node_config"]
            cls = factory.get_model_cls(_dtype)
            self.node_2cls_dict[_name] = {
                "name": _name,
                "input": _input,
                "output": [],
                "cls": cls(_config),
                "dag_is_input": _dag_is_input,
                "dag_is_output": _dag_is_output,
            }
        for node in self.model_dag:
            _name = node["node_name"]
            _input = node["node_input"].split(",")
            for _in in _input:
                inp = self.node_2cls_dict.get(_in, None)
                if inp is None:
                    raise ValueError(f"`node input:{_in} not in dag`")
                inp["output"].append(f"{_name}")

    def build_dag(self):
        queue = [self.node_2cls_dict["start"]]
        node_2out_dict = {}
        dag_input = {}
        dag_output = {}
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.pop(0)
                _name = node["name"]
                _input = node["input"]
                _output = node["output"]
                _cls = node["cls"]
                _dag_is_input = node["dag_is_input"]
                _dag_is_output = node["dag_is_output"]

                if _name == "start":
                    for _out in _output:
                        node_2out_dict[f"{_name}->{_out}"] = None
                        queue.append(self.node_2cls_dict[_out])
                    continue

                if len(_input) == 1:
                    res = _cls(node_2out_dict[f"{_input[0]}->{_name}"])
                    for _out in _output:
                        node_2out_dict[f"{_name}->{_out}"] = res
                        queue.append(self.node_2cls_dict[_out])
                    if _dag_is_input:
                        dag_input[_name] = res
                    if _dag_is_output:
                        dag_output[_name] = res
                    continue

                for _in in _input:
                    tmp = []
                    for _in in _input:
                        res = _cls(node_2out_dict[f"{_in}->{_name}"])
                        tmp.append(res)
                        if _dag_is_input:
                            dag_input[f"{_in}->{_name}"] = res
                        if _dag_is_output:
                            dag_output[f"{_in}->{_name}"] = res

                    for _out in _output:
                        node_2out_dict[f"{_name}->{_out}"] = tmp
                        queue.append(self.node_2cls_dict[_out])

        self._model = tf.keras.Model(dag_input, dag_output)
        self._model.summary()

    def handler(self):
        self.instantiate()
        self.build_dag()

        return self._model


def product_StandardModelOptions(model_dag):
    cls = StandardModel(model_dag).handler()
    _conf_name = "mcf"

    @factory.register_model_cls(_conf_name)
    class StandardModelOptionsNode(ModelOptions):
        conf_name = "mcf"
        model = cls

    logging.info(f"\nregister_model_cls: {_conf_name} is success\n")
