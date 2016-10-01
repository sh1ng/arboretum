# arboretum -  Gradient Boosting on GPU

## Dependencies
* Python 2.7
* Cuda toolkit 8(there's no cuda 8 sprecifiec features are used so it should also work with cuda 7, but it has never been tested).
* Cuda thrust - it's a part of cuda toolkit so it should work out of box.
* Cuda cub https://github.com/NVlabs/cub located as a sibling of the project. Change location in the makefile if require. 

## Installation process is similar to xgboost
* $ make 
* $ cd python-wrapper
* $ sudo python setup.py install
