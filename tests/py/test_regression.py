import arboretum
import numpy as np
from sklearn.datasets import load_boston
import json
import pytest


def run_regression(depth, true_values, trees=1):
    boston = load_boston()
    n = 10

    # create data matrix
    data = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
    y = boston.target[0:n]

    config = json.dumps({'objective': 0,
                         'internals':
                         {
                             'double_precision': True
                         },
                         'tree':
                         {
                             'eta': 1.0,
                             'max_depth': depth,
                             'gamma': 0.0,
                             'min_child_weight': 2,
                             'min_leaf_size': 2,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0
                         }})
    model = arboretum.Garden(config, data)
    for _ in range(trees):
        model.grow_tree()
    pred = model.predict(data)
    assert np.allclose(pred, true_values)


def test_single_tree_depth_2(): run_regression(2, [21.833334, 21.833334, 33.25,     33.25,     33.25,     33.25,     21.833334,
                                                   21.833334, 21.833334, 21.833334])


def test_single_tree_depth_3(): run_regression(3, [23.9,      23.9,      35.45,     31.050001, 35.45,     31.050001, 23.9,
                                                   23.9,      17.7,      17.7])


def test_single_tree_depth_4(): run_regression(4, [22.8,      22.8,      35.45,     31.050001, 35.45,     31.050001, 25,
                                                   25.,      17.7,      17.7])


def test_2trees_depth_2(): run_regression(2, [23.72778,  23.72778,  30.408333, 35.144444, 35.144444, 30.408333, 23.72778,
                                              23.72778,  18.991667, 18.991667], 2)


def test_2trees_depth_3(): run_regression(3, [23.575,    21.575,    35.125,    33.15,     37.550003, 28.725002, 23.566666,
                                              26,       17.366667, 17.366667], 2)


def test_2trees_depth_4(): run_regression(4, [23.5,      23.5,      35.2,      32.600002, 37,       28.825,    22.775,
                                              25.7,      17.45,     17.45], 2)


if __name__ == "__main__":
    pass
