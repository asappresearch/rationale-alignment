"""
setup.py
"""

from setuptools import setup, find_packages
from typing import Dict
import os


NAME = "rationale-alignment"
AUTHOR = "ASAPP Inc."
EMAIL = "liliyu@asapp.com"
DESCRIPTION = (
    "Pytorch based library for ACL2020 paper about rationalizing text matching."
)


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def required():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name=NAME,
    version="0.0.1",
    description=DESCRIPTION,
    # Author information
    author=AUTHOR,
    author_email=EMAIL,
    license="MIT",
    # What is packaged here.
    packages=find_packages(),
    install_requires=required(),
    python_requires=">=3.6.1",
    zip_safe=True,
)
