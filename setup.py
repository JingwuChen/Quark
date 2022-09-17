# -*- coding: utf-8 -*-
# @Time   : 2021/10/27 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = ['h5py==2.10.0', 'requests']

setuptools.setup(
    name="quark",
    version="0.0.1",
    author="merlin, oliver, guitao, xudong",
    author_email="merlinzhu@lexin.com",
    description="easy-to-use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="None",
    packages=setuptools.find_packages(exclude=["tests", "doc", "example"]),
    python_requires=">=3.6",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "cpu": ["tensorflow>=2.9.0"],
        "gpu": ["tensorflow-gpu>=2.9.0"],
    },
    entry_points={},
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['train framework', 'xgb', 'deep learning', 'tensorflow'],
)
