#!/usr/bin/env python

import os
from setuptools import setup, find_packages


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf8') as f:
        return f.read()


setup(
    name="SurvivalEVAL",
    version="0.1.dev0",
    packages=find_packages(),
    author="Shi-ang Qi",
    author_email="shiang@ualberta.ca",
    description="The most comprehensive Python package for evaluating survival analysis models.",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url="https://github.com/shi-ang/SurvivalEVAL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

'''
TODO:
1. install setuptools 

python -m pip install --user --upgrade setuptools wheel

2. run the setup.py command

python setup.py sdist bdist_wheel

Test to verify it works

python -m pip install --index-url https://test.pypi.org/simple/ --no-deps Your-Package-Name

'''