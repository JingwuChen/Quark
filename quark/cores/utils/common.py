# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
import os
import random


# 获得随机值
def get_random_integer():
    return random.randint(0, (1 << 31) - 1)


# 判断文件夹
def is_dir(files):
    return os.path.isdir(files)


# 判断文件
def is_file(files):
    return os.path.isfile(files)


# 获取文件夹下的所有文件
def get_files_from_dir(files):
    if is_file(files):
        return [files]
    if is_dir(files):
        files_list = []
        for file_name in os.listdir(files):
            tmp = files + "/" + file_name
            if os.access(tmp, os.F_OK) and os.access(tmp, os.R_OK):
                files_list.append(tmp)
            else:
                print('%s is not exist or not read' % tmp)
        return files_list
