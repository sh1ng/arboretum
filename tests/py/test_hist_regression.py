import arboretum
import numpy as np
from sklearn.datasets import load_boston
import json
import pytest


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.mean(diff))


def run_regression(depth, true_values, trees=1, double_precision=True, use_hist_trick=True, upload_features=True):
    boston = load_boston()
    n = 10

    # create data matrix
    data = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
    y = boston.target[0:n]

    config = json.dumps({'objective': 0,
                         'method': 1,
                         'hist_size': 12,
                         'internals':
                         {
                             'double_precision': double_precision,
                             'compute_overlap': 2,
                             'use_hist_subtraction_trick': use_hist_trick,
                             'upload_features': upload_features,
                         },
                         'tree':
                         {
                             'eta': 0.99999,
                             'max_depth': depth,
                             'gamma': 0.0,
                             'min_child_weight': 2,
                             'min_leaf_size': 2,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0,
                         }})
    model = arboretum.Garden(config, data)
    for _ in range(trees):
        model.grow_tree()
    data = arboretum.DMatrix(boston.data[0:n])

    pred = model.predict(data)
    print(pred)
    print(true_values)
    print(boston.target[0:n])
    print('rmse:', rmse(pred, true_values))
    assert np.allclose(pred, true_values)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [21.833334, 21.833334, 33.25,     33.25,     33.25,     33.25,     21.833334,
                                                                                                    21.833334, 21.833334, 21.833334],
                                                                                                double_precision=double_precision,
                                                                                                use_hist_trick=use_hist_trick,
                                                                                                upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_3(double_precision, use_hist_trick, upload_features): run_regression(3, [23.9,      23.9,      35.45,     31.050001, 35.45,     31.050001, 23.9,
                                                                                                    23.9,      17.7,      17.7],
                                                                                                double_precision=double_precision,
                                                                                                use_hist_trick=use_hist_trick,
                                                                                                upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_4(double_precision, use_hist_trick, upload_features): run_regression(4, [22.8,      22.8,      35.45,     31.050001, 35.45,     31.050001, 25,
                                                                                                    25.,      17.7,      17.7],
                                                                                                double_precision=double_precision,
                                                                                                use_hist_trick=use_hist_trick,
                                                                                                upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [23.72778,  23.72778,  30.408333, 35.144444, 35.144444, 30.408333, 23.72778,
                                                                                               23.72778,  18.991667, 18.991667], 2,
                                                                                           double_precision=double_precision,
                                                                                           use_hist_trick=use_hist_trick,
                                                                                           upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_3(double_precision, use_hist_trick, upload_features): run_regression(3, [23.575,    21.575,    35.125,    33.15,     37.550003, 28.725002, 23.566666,
                                                                                               26,       17.366667, 17.366667], 2,
                                                                                           double_precision=double_precision,
                                                                                           use_hist_trick=use_hist_trick,
                                                                                           upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_4(double_precision, use_hist_trick, upload_features): run_regression(4, [23.5,      23.5,      35.2,      32.600002, 37,       28.825,    22.775,
                                                                                               25.7,      17.45,     17.45], 2,
                                                                                           double_precision=double_precision,
                                                                                           use_hist_trick=use_hist_trick,
                                                                                           upload_features=upload_features)


if __name__ == "__main__":
    test_single_tree_depth_3(True, True, True, True)
    # test_2trees_depth_4(True)
