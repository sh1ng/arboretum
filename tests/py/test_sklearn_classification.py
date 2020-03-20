import arboretum
import numpy as np
from sklearn.datasets import load_iris
import json
import pytest
import utils


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("method", ['hist', 'exact'])
@pytest.mark.parametrize("hist_size", [255, 256, 511, 512, 1023])
def test_single_tree(double_precision, method, hist_size, y_pred=[[0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.5, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.42555746, 0.59668386,
                                                                   0.5, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.5,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.5,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
                                                                   0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.40131232, 0.42555746,
                                                                   0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.59668386, 0.40131232,
                                                                   0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.42555746,
                                                                   0.42555746, 0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.5,
                                                                   0.40131232, 0.42555746, 0.40131232, 0.40131232, 0.40131232, 0.40131232,
                                                                   0.40131232, 0.40131232, 0.40131232, 0.5, 0.40131232, 0.40131232,
                                                                   0.40131232, 0.5, 0.5, 0.40131232, 0.40131232, 0.40131232,
                                                                   0.42555746, 0.40131232, 0.40131232, 0.40131232, 0.42555746, 0.40131232,
                                                                   0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.42555746]]):
    iris = load_iris()
    n = 10000
    y = iris.target != 2

    data = arboretum.DMatrix(iris.data[0:n], y=y)

    model = arboretum.ArboretumClassifier(max_depth=2, learning_rate=0.2, n_estimators=1,
                                          verbosity=0,
                                          gamma_absolute=0.0,
                                          gamma_relative=0.0,
                                          min_child_weight=2.0,
                                          min_leaf_size=0,
                                          max_leaf_weight=0,
                                          colsample_bytree=1.0,
                                          colsample_bylevel=1.0,
                                          l1=0,
                                          l2=0,
                                          scale_pos_weight=0.5,
                                          initial_y=0.5,
                                          seed=0,
                                          double_precision=double_precision,
                                          method=method,
                                          hist_size=hist_size)
    model.fit(data, y)
    assert np.allclose(model.predict(data), y_pred)