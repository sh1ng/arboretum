# arboretum -  Gradient Boosting on GPU

## Dependencies
* Python 2.7
* Cuda toolkit 8(there's no cuda 8 sprecifiec features are used so it should also work with cuda 7, but it has never been tested).
* Cuda thrust - it's a part of cuda toolkit so it should work out of box.
* Cuda cub https://github.com/NVlabs/cub as submodule. 

## Linux
* gcc-5

## MacOS
* clang-omp++ can be install by brew

## Installation process is similar to xgboost
* git clone --recursive https://github.com/sh1ng/arboretum.git
* $ make 
* $ cd python-wrapper
* $ sudo python setup.py install
