# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2022/9/3|下午 03:12
# @Moto   : Knowledge comes from decomposition

import abc
import common.text_normalize as text_normalize
from typing import List, Union, Optional, Tuple


class PluginManager(metaclass=abc.ABCMeta):

    def __init__(self):
        self.plugins = dict()

    # 获取插件
    def get(self, plugin_name: str):
        return self.plugins[plugin_name]

    # 插件管理核心操作
    def handler(self):
        pass

    # 注册
    def register(self, plugin_name_list: List[object]):
        for plugin in plugin_name_list:
            self.plugins[plugin.name] = plugin
            print(f'注册插件：{plugin.name}')


class PluginBase(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    # 设置输入
    def set_input(self):
        pass

    # 设置输出
    def set_output(self):
        pass

    # 插件核心操作
    def handler(self, input_data):
        pass

    # def call(self, input_data):
    #     self.set_input()
    #     self.set_output()
    #     return self.handler(input_data)


class TextNormalization(PluginBase):
    """
    文本规范化插件
    - 1. 全角转半角
    - 2. 大写字母转为小写字母
    - 3. 转为简体
    """

    def __init__(self) -> None:
        self.name = 'text_normal'
        super().__init__()

    def handler(self, input_data):
        text = text_normalize.full_to_half(input_data)
        text = text_normalize.upper2lower(text)
        text = text_normalize.traditional2simplified(text)
        print(text)

        return text


# 已全局调用进行使用
Pluginmanager = PluginManager()

if __name__ == '__main__':
    text_normal = TextNormalization()
    text_normal.handler('我是中国人Chinese，我爱我的祖国！')
    Pluginmanager.register([text_normal])
    Pluginmanager.get('text_normal').handler(input_data='我是中国人Chinese，我爱我的祖国！')

    pass
