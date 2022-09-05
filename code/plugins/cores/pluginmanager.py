# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2022/9/3|下午 03:12
# @Moto   : Knowledge comes from decomposition

import abc
from typing import List, Union, Optional, Tuple


class PluginManager(metaclass=abc.ABCMeta):

    def __init__(self):
        self.plugins = dict()

    # 获取插件
    def get(plugin_name: str):
        pass

    # 插件管理核心操作
    def handler():
        pass

    # 注册
    def register(plugin_name_list: List[str]):
        
        pass


class PluginBase(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    # 设置输入
    def set_input():
        pass

    # 设置输出
    def set_output():
        pass

    # 插件核心操作
    def handler():
        pass


# 已全局调用进行使用
Pluginmanager = PluginManager()
