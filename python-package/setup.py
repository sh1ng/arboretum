# pylint: disable=invalid-name, exec-used
"""Setup arboretum package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages
#import subprocess
sys.path.insert(0, '.')

setup(name='arboretum',
      version='0.1',
      description='',
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Vladimir Ovsyannikov',
      maintainer_email='vladimir.ovsyannikov@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=True,
      data_files=[('arboretum', [ '../python-wrapper/arboretum_wrapper.so'])],
      url='https://github.com/sh1ng/arboretum')