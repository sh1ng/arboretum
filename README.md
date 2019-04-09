# arboretum -  Gradient Boosting on GPU

## Dependencies
* Python 2.7 or Python 3
* Cuda toolkit 7+
* Cuda thrust - it's a part of cuda toolkit so it should work out of box
* Cuda cub https://github.com/NVlabs/cub as a submodule
* JSON for Modern C++ https://github.com/nlohmann/json as a submodule

## Linux
* gcc-5


## Installation process
* git clone --recursive https://github.com/sh1ng/arboretum.git
* Edit CmakeList.txt to specify [computational capabilities of your device](https://developer.nvidia.com/cuda-gpus) major\*100+minor\*10
* $ mkdir build && cd build && cmake .. && make -j
* $ cd ../python-package
* $ sudo python setup.py install
