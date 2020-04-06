# arboretum - Gradient Boosting implementation with focus on overcoming GRAM size limit

## Installation wheel package 
```
pip install arboretum
```

[Benchmark results](https://docs.google.com/spreadsheets/d/1XAqtHsch0u6-oiknHGbSjTgkIC43NVxaZjHKDcVmRW8/edit?usp=sharing)

## Dependencies
* Python 2.7 or Python 3
* Cuda toolkit 7+
* Cuda cub https://github.com/NVlabs/cub as a submodule
* JSON for Modern C++ https://github.com/nlohmann/json as a submodule

## Installation from source
* git clone --recursive https://github.com/sh1ng/arboretum.git
* $ mkdir build && cd build && cmake .. && make -j && cd .. && make wheel
* $ sudo python -m pip install python-package/dist/arboretum*.whl 
