# pylint: disable=invalid-name, exec-used
"""Setup arboretum package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages
sys.path.insert(0, '.')

setup(name='arboretum',
      version='0.1a5',
      description='Gradient Boosting powered by GPU(NVIDIA CUDA)',
      long_description='Gradient Boosting implementation with focus on overcoming GRAM size limit',
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Vladimir Ovsyannikov',
      maintainer_email='vladimir.ovsyannikov@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=True,
      url='https://github.com/sh1ng/arboretum',
      python_requires='>=3.5',
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Development Status :: 3 - Alpha',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: C++',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],)
