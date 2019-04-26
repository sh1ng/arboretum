import arboretum
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import json
from scipy.sparse import csc_matrix
import pytest


def convert2category(x):
    unq = np.unique(x)
    return np.searchsorted(unq, x).astype(np.uint32)


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.sum(diff))


@pytest.mark.parametrize("depth", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("trees", [1, 10, 20, 50])
def test_category(depth, trees):
    # load test data
    boston = load_boston()

    # features with only 2 unique values, so it can be converted to category
    categoties = [3]
    data_categories = []
    for item in categoties:
        data_categories.append(convert2category(boston.data[:, item]))

    data_categories = np.stack(data_categories, axis=-1)

    data_source = boston.data[:, 4:5]

    # create data matrix
    data = arboretum.DMatrix(
        data_source, data_category=data_categories, y=boston.target)
    y = boston.target

    config = json.dumps({'objective': 0,
                         'internals':
                         {
                             'double_precision': True,
                             'compute_overlap': 1
                         },
                         'verbose':
                         {
                             'gpu': True
                         },
                         'tree':
                         {
                             'eta': 0.5,
                             'max_depth': depth,
                             'gamma': 0.0,
                             'min_child_weight': 1,
                             'min_leaf_size': 1,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0
                         }})

    # init model
    #model = arboretum.Garden('reg:linear', data, 6, 2, 1, 0.5)
    model = arboretum.Garden(config, data)

    # grow trees
    for i in range(trees):
        model.grow_tree()

    pred = model.get_y(data)

    # predict on train data set
    pred = model.predict(data)

    data_source = boston.data[:, 3:5]

    # create data matrix
    data = arboretum.DMatrix(data_source, y=boston.target)
    y = boston.target

    config = json.dumps({'objective': 0,
                         'internals':
                         {
                             'double_precision': True,
                             'compute_overlap': 1
                         },
                         'verbose':
                         {
                             'gpu': True
                         },
                         'tree':
                         {
                             'eta': 0.5,
                             'max_depth': depth,
                             'gamma': 0.0,
                             'min_child_weight': 1,
                             'min_leaf_size': 1,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0
                         }})

    model = arboretum.Garden(config, data)

    # grow trees
    for i in range(trees):
        model.grow_tree()

    pred1 = model.get_y(data)

    assert np.allclose(pred, pred1)

    # predict on train data set
    pred1 = model.predict(data)

    assert np.allclose(pred, pred1)
