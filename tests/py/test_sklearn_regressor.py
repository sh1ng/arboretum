import arboretum
import numpy as np
from sklearn.datasets import load_boston
import json
import pytest
import utils


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("method", ['hist', 'exact'])
@pytest.mark.parametrize("hist_size", [12, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023])
def test_single_tree(double_precision, method, hist_size, y_pred=[[21.833334, 21.833334, 33.25,     33.25,     33.25,     33.25,     21.833334,
                                                                   21.833334, 21.833334, 21.833334]]):
    boston = load_boston()
    n = 10
    data = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
    y = boston.target[0:n]

    model = arboretum.ArboretumRegression(max_depth=1, learning_rate=0.99999, n_estimators=1,
                                          verbosity=0,
                                          gamma_absolute=0.0,
                                          gamma_relative=0.0,
                                          min_child_weight=1.0,
                                          min_leaf_size=1,
                                          max_leaf_weight=0,
                                          colsample_bytree=1.0,
                                          colsample_bylevel=1.0,
                                          l1=0,
                                          l2=0,
                                          scale_pos_weight=1.0,
                                          initial_y=0.5,
                                          seed=0,
                                          double_precision=double_precision,
                                          method=method,
                                          hist_size=hist_size)
    model.fit(data, y)
    pred = model.predict(data)
    print(pred)
    print(y_pred)
    assert np.allclose(pred, y_pred)
