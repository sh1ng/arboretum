# arboretum -  Gradient Boosting on GPU

## Dependencies
* Python 2.7 or Python 3
* Cuda toolkit 8 or 7
* Cuda thrust - it's a part of cuda toolkit so it should work out of box
* Cuda cub https://github.com/NVlabs/cub as a submodule
* JSON for Modern C++ https://github.com/nlohmann/json as a submodule

## Linux
* gcc-5

## MacOS
* clang-omp++ can be install by brew

## Installation process
* git clone --recursive https://github.com/sh1ng/arboretum.git
* $ make 
* $ cd python-wrapper
* $ sudo python setup.py install
