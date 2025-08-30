#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from SurvivalEVAL.version import __version__

exec(compile(open("SurvivalEVAL/version.py").read(), "SurvivalEVAL/version.py", "exec"))


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf8') as f:
        return f.read()


setup(
    name="SurvivalEVAL",
    version=__version__,
    packages=find_packages(),
    author="Shi-ang Qi, Weijie Sun",
    author_email="shiang@ualberta.ca, weijie2@ualberta.ca",
    description="The most comprehensive Python package for evaluating survival analysis models.",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url="https://github.com/shi-ang/SurvivalEVAL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
    keywords="survival analysis, evaluation, metrics, survivaleval",
    license="MIT",
    install_requires=read('requirements.txt').splitlines(),
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