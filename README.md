# arboretum - Gradient Boosting implementation with focus on overcoming GRAM size limit

## Installation wheel package 
```
pip install arboretum
```

## Dependencies
* Python 2.7 or Python 3
* Cuda toolkit 7+
* Cuda cub https://github.com/NVlabs/cub as a submodule
* JSON for Modern C++ https://github.com/nlohmann/json as a submodule

## Installation from source
* git clone --recursive https://github.com/sh1ng/arboretum.git
* $ mkdir build && cd build && cmake .. && make -j && cd .. && make wheel
* $ sudo python -m pip install python-package/dist/arboretum*.whl 
