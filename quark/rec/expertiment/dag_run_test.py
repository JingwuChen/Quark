# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../../..")
print("==================")
print(sys.path)
print("==================")
import yaml
from quark.cores import factory
from absl import flags, app

flags.DEFINE_string('file_path', 'params.yaml',
                    'List of paths to the config files.')
FLAGS = flags.FLAGS


def main(_):
    with open(FLAGS.file_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    _cls = factory.get_exp_cls("ExperimentManager")
    exp01 = _cls(params)
    exp01.handler()


if __name__ == '__main__':
    app.run(main)
