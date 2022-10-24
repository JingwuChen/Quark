# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import abc
from threading import Thread
from queue import Queue
from quark.cores import ExperimentConfig
from quark.cores import product_PrintDataSetOptions
from quark.cores import product_StandardInputerOptions
from quark.cores import product_StandardProcessorOptions
from quark.cores import product_StandardTrainerOptions
from quark.cores import product_StandardEvaluatorOptions
from quark.cores import product_StandardModelOptions
from quark.cores import product_StandardSaveModel
from quark.cores import factory
from absl import logging
from typing import Dict
# Global Var
NODE_NAME = "node_name"
NODE_DTYPE = "node_dtype"
NODE_INPUT = "node_input"
NODE_OUTPUT = "node_output"
NODE_CONFIG = "node_config"
DAG_START = "start"
DAG_END = "end"

# Global Node
STANDARDINPUTER = "StandardInputer"
PRINTDATASET = "PrintDataSet"
STANDARDPROCESSOR = "StandardProcessor"
STANDARDTRAINER = "StandardTrainer"
STANDARDEVALUATOR = "StandardEvaluator"
STANDARDSAVEMODEL = "StandardSaveModel"


class ExperimentManagerOptions(ExperimentConfig):
    conf_name: str = None
    task_dag: Dict = None
    model_dag: Dict = None
    runtime: Dict = None


@factory.register_exp_cls('ExperimentManager')
class ExperimentManager(metaclass=abc.ABCMeta):

    def __init__(self, params):
        self.task_dag = params["task_dag"]
        self.model_dag = params["model_dag"]
        self.runtime = params["runtime"]
        self.dag_dict = {}

    def handler(self):
        self._check_yaml()
        self._registry_conf()
        self._build_dag()
        self._build_queue()
        self._start()

    def _check_yaml(self):
        for node in self.task_dag:
            if node.get(NODE_NAME) is None:
                raise ValueError(f'`{NODE_NAME}` can be specified.')
            if node.get(NODE_DTYPE) is None:
                raise ValueError(f'`{NODE_DTYPE}` can be specified.')
            if node.get(NODE_INPUT) is None:
                raise ValueError(f'`{NODE_INPUT}` can be specified.')
            if node.get(NODE_CONFIG) is None:
                raise ValueError(f'`{NODE_CONFIG}` can be specified.')

    def _registry_model(self):
        product_StandardModelOptions(self.model_dag)

    def _registry_conf(self):

        for node in self.task_dag:

            config = node[NODE_CONFIG]
            dtype = node[NODE_DTYPE]
            if dtype == STANDARDINPUTER:
                product_StandardInputerOptions(config)
            if dtype == PRINTDATASET:
                product_PrintDataSetOptions(config)
            if dtype == STANDARDPROCESSOR:
                product_StandardProcessorOptions(config)
            if dtype == STANDARDTRAINER:
                product_StandardTrainerOptions(config)
            if dtype == STANDARDEVALUATOR:
                product_StandardEvaluatorOptions(config)
            if dtype == STANDARDSAVEMODEL:
                product_StandardSaveModel(config)
        self._registry_model()

    def _build_dag(self):
        self.exp_dag = {}
        for node in self.task_dag:
            _name = node[NODE_NAME]
            _dtype = node[NODE_DTYPE]
            _input = node[NODE_INPUT].split(",")
            _output = []
            _conf_name = node[NODE_CONFIG]["conf_name"]
            self.exp_dag[_name] = {
                "name": _name,
                "dtype": _dtype,
                "input": _input,
                "output": _output,
                "conf_name": _conf_name,
            }
            for _in in _input:
                if self.exp_dag.get(_in, None) is not None:
                    self.exp_dag[_in]["output"].append(_name)

        for ex, info in self.exp_dag.items():
            logging.info(f"\nkey: {ex}, info: {info}\n")

    def _build_queue(self):
        self.queue_pool = {}
        for ex, info in self.exp_dag.items():
            for ou in info["output"]:
                self.queue_pool[f"{ex}/{ou}"] = Queue(maxsize=1)

    def _run_node(self, node):
        _name = node["name"]
        _dtype = node["dtype"]
        _input = node["input"]
        _output = node["output"]
        _conf_name = node["conf_name"]
        # if _name == "val_input":
        #     sleep(5)
        logging.info(f'\nThreading: {_name} start ...\n')
        if _input[0] == DAG_START:
            _class = factory.get_task_cls(_dtype)
            _config = factory.get_task_cls(_conf_name)
            result = _class(_config).handler()
            for _ou in _output:
                _oq = self.queue_pool[f"{_name}/{_ou}"]
                _oq.put(result)
        else:
            values = []
            for _in in _input:
                _iq = self.queue_pool[f"{_in}/{_name}"]
                value = _iq.get(True)
                values.append(value)
            _class = factory.get_task_cls(_dtype)
            _config = factory.get_task_cls(_conf_name)
            result = _class(_config).handler(values)
            for _ou in _output:
                _oq = self.queue_pool[f"{_name}/{_ou}"]
                _oq.put(result)
        logging.info(f'\nThreading: {_name} end ...\n')

    def _start(self):
        for _, node in self.exp_dag.items():
            p = Thread(target=self._run_node, args=(node, ))
            p.start()


# def product_ExperimentManager(conf):
#     if conf.get("reuse", False):
#         return
#     _conf_name = conf.get("conf_name")
#     if _conf_name is None:
#         raise ValueError("`conf_name` can be specified.")
#     _task_dag = conf.get("task_dag", None)
#     if _task_dag is None:
#         raise ValueError("`task_dag` can be specified.")

#     _model_dag = conf.get("model_dag", None)
#     if _model_dag is None:
#         raise ValueError("`model_dag` can be specified.")

#     _runtime = conf.get("runtime", None)

#     @factory.register_task_cls(_conf_name)
#     class ExperimentManagerOptionsNode(ExperimentManagerOptions):
#         conf_name = _conf_name
#         task_dag: Dict = _task_dag
#         model_dag: Dict = _model_dag
#         runtime: Dict = _runtime

#     logging.info(f"\nregister_exp_cls: {_conf_name} is success\n")
