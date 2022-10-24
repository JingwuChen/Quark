# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A parameter dictionary class."""

import copy

import six
import tensorflow as tf
import yaml


class ParamsDict(object):
    """A hyperparameter container class."""
    RESERVED_ATTR = ['_locked', '_restrictions']

    def __init__(self, default_params=None, restrictions=None):
        """Instantiate a ParamsDict.
        """
        self._locked = False
        self._restrictions = []
        if restrictions:
            self._restrictions = restrictions
        if default_params is None:
            default_params = {}
        self.override(default_params, is_strict=False)

    def _set(self, k, v):
        if isinstance(v, dict):
            self.__dict__[k] = ParamsDict(v)
        else:
            self.__dict__[k] = copy.deepcopy(v)

    def __setattr__(self, k, v):
        if k not in ParamsDict.RESERVED_ATTR:
            if k not in self.__dict__.keys():
                raise KeyError('The key `%{}` does not exist. '
                               'To extend the existing keys, use '
                               '`override` with `is_strict` = True.'.format(k))
            if self._locked:
                raise ValueError('The ParamsDict has been locked. '
                                 'No change is allowed.')
        self._set(k, v)

    def __getattr__(self, k):
        if k not in self.__dict__.keys():
            raise AttributeError('The key `{}` does not exist. '.format(k))
        return self.__dict__[k]

    def __contains__(self, key):
        return key in self.__dict__

    def get(self, key, value=None):
        return self.__dict__.get(key, value)

    def __delattr__(self, k):
        if k in ParamsDict.RESERVED_ATTR:
            raise AttributeError(
                'The key `{}` is reserved. No change is allowes. '.format(k))
        if k not in self.__dict__.keys():
            raise AttributeError('The key `{}` does not exist. '.format(k))
        if self._locked:
            raise ValueError(
                'The ParamsDict has been locked. No change is allowed.')
        del self.__dict__[k]

    def override(self, override_params, is_strict=True):
        if self._locked:
            raise ValueError(
                'The ParamsDict has been locked. No change is allowed.')
        if isinstance(override_params, ParamsDict):
            override_params = override_params.as_dict()
        self._override(override_params, is_strict)  # pylint: disable=protected-access

    def _override(self, override_dict, is_strict=True):
        """The implementation of `override`."""
        for k, v in six.iteritems(override_dict):
            if k in ParamsDict.RESERVED_ATTR:
                raise KeyError('The key `%{}` is internally reserved. '
                               'Can not be overridden.')
            if k not in self.__dict__.keys():
                if is_strict:
                    raise KeyError(
                        'The key `{}` does not exist. '
                        'To extend the existing keys, use '
                        '`override` with `is_strict` = False.'.format(k))
                else:
                    self._set(k, v)
            else:
                if isinstance(v, dict):
                    self.__dict__[k]._override(v, is_strict)  # pylint: disable=protected-access
                elif isinstance(v, ParamsDict):
                    self.__dict__[k]._override(v.as_dict(), is_strict)  # pylint: disable=protected-access
                else:
                    self.__dict__[k] = copy.deepcopy(v)

    def lock(self):
        """Makes the ParamsDict immutable."""
        self._locked = True

    def as_dict(self):
        """Returns a dict representation of ParamsDict.

        For the nested ParamsDict, a nested dict will be returned.
        """
        params_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if k not in ParamsDict.RESERVED_ATTR:
                if isinstance(v, ParamsDict):
                    params_dict[k] = v.as_dict()
                else:
                    params_dict[k] = copy.deepcopy(v)
        return params_dict


def read_yaml_to_params_dict(file_path: str):
    """Reads a YAML file to a ParamsDict."""
    with tf.io.gfile.GFile(file_path, 'r') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)
        return ParamsDict(params_dict)


def save_params_dict_to_yaml(params, file_path):
    """Saves the input ParamsDict to a YAML file."""
    with tf.io.gfile.GFile(file_path, 'w') as f:

        def _my_list_rep(dumper, data):
            # u'tag:yaml.org,2002:seq' is the YAML internal tag for sequence.
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq',
                                             data,
                                             flow_style=True)

        yaml.add_representer(list, _my_list_rep)
        yaml.dump(params.as_dict(), f, default_flow_style=False)


def override_params_dict(params, dict_or_string_or_yaml_file, is_strict):
    if isinstance(dict_or_string_or_yaml_file, six.string_types):
        params_dict = yaml.load(dict_or_string_or_yaml_file,
                                Loader=yaml.FullLoader)
        params.override(params_dict, is_strict)
    else:
        raise ValueError('Unknown input type to parse.')
    return params
