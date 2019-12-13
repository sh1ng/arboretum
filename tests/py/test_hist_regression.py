import arboretum
import numpy as np
from sklearn.datasets import load_boston
import json
import pytest
import utils


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.mean(diff))


def run_regression(depth, true_values, true_model, trees=1, double_precision=True, use_hist_trick=True, upload_features=True):
    boston = load_boston()
    n = 10

    # create data matrix
    data = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
    y = boston.target[0:n]

    config = {'objective': 0,
                         'method': 1,
                         'internals':
                         {
                             'double_precision': double_precision,
                             'compute_overlap': 2,
                             'use_hist_subtraction_trick': use_hist_trick,
                             'upload_features': upload_features,
                             'hist_size': 12,
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
                         }}
    model = arboretum.train(config, data, trees)
    data = arboretum.DMatrix(boston.data[0:n])

    pred = model.predict(data)
    print(pred)
    print(true_values)
    print(boston.target[0:n])
    print('rmse:', rmse(pred, true_values))
    assert np.allclose(pred, true_values)
    model_json_str = model.dump()
    # print(model_json_str)
    model = json.loads(model_json_str)
    print(type(true_model), type(model))
    utils.assert_model(true_model, model)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [21.833334, 21.833334, 33.25,     33.25,     33.25,     33.25,     21.833334,
                                                                                                    21.833334, 21.833334, 21.833334],
                                                                                                json.loads(
    '{"configuration":{"internals":{"compute_overlap":2,"double_precision":true,"hist_size":12,"seed":0,"upload_features":true,"use_hist_subtraction_trick":true},"method":1,"objective":0,"tree":{"alpha":0.0,"colsample_bylevel":1.0,"colsample_bytree":1.0,"eta":0.9999899864196777,"gamma_absolute":0.0,"gamma_relative":0.0,"initial_y":0.5,"labels_count":1,"lambda":0.0,"max_depth":2,"max_leaf_weight":0.0,"min_child_weight":2.0,"min_leaf_size":2,"scale_pos_weight":0.5},"verbose":{"booster":false,"data":false,"gpu":false}},"model":[{"nodes":[{"fid":6,"id":0,"leaf":false,"left":1,"right":2,"threshold":63.14999771118164},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[32.749671936035156,21.333120346069336]}]}'),
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_3(double_precision, use_hist_trick, upload_features): run_regression(3, [23.9,      23.9,      35.45,     31.050001, 35.45,     31.050001, 23.9,
                                                                                                    23.9,      17.7,      17.7],
                                                                                                json.loads(
    '{"configuration": {"internals": {"compute_overlap": 2, "double_precision": false, "hist_size": 12, "seed": 0, "upload_features": false, "use_hist_subtraction_trick": false}, "method": 1, "objective": 0, "tree": {"alpha": 0.0, "colsample_bylevel": 1.0, "colsample_bytree": 1.0, "eta": 0.9999899864196777, "gamma_absolute": 0.0, "gamma_relative": 0.0, "initial_y": 0.5, "labels_count": 1, "lambda": 0.0, "max_depth": 3, "max_leaf_weight": 0.0, "min_child_weight": 2.0, "min_leaf_size": 2, "scale_pos_weight": 0.5}, "verbose": {"booster": false, "data": false, "gpu": false}}, "model": [{"nodes": [{"fid": 6, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 63.14999771118164}, {"fid": 5, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": 7.072500228881836}, {"fid": 0, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 0.15729498863220215}, {"id": 3, "idx": 0, "leaf": true}, {"id": 4, "idx": 1, "leaf": true}, {"id": 5, "idx": 2, "leaf": true}, {"id": 6, "idx": 3, "leaf": true}], "weights": [30.54969596862793, 34.94964599609375, 23.39976692199707, 17.199825286865234]}]}'),
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_4(double_precision, use_hist_trick, upload_features): run_regression(4, [22.8,      22.8,      35.45,     31.050001, 35.45,     31.050001, 25,
                                                                                                    25.,      17.7,      17.7],
                                                                                                json.loads(
    '{"configuration": {"internals": {"compute_overlap": 2, "double_precision": false, "hist_size": 12, "seed": 0, "upload_features": false, "use_hist_subtraction_trick": false}, "method": 1, "objective": 0, "tree": {"alpha": 0.0, "colsample_bylevel": 1.0, "colsample_bytree": 1.0, "eta": 0.9999899864196777, "gamma_absolute": 0.0, "gamma_relative": 0.0, "initial_y": 0.5, "labels_count": 1, "lambda": 0.0, "max_depth": 4, "max_leaf_weight": 0.0, "min_child_weight": 2.0, "min_leaf_size": 2, "scale_pos_weight": 0.5}, "verbose": {"booster": false, "data": false, "gpu": false}}, "model": [{"nodes": [{"fid": 6, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 63.14999771118164}, {"fid": 5, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": 7.072500228881836}, {"fid": 0, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 0.15729498863220215}, {"fid": 0, "id": 3, "leaf": false, "left": 7, "right": 8, "threshold": null}, {"fid": 0, "id": 4, "leaf": false, "left": 9, "right": 10, "threshold": null}, {"fid": 0, "id": 5, "leaf": false, "left": 11, "right": 12, "threshold": 0.02858000062406063}, {"fid": 0, "id": 6, "leaf": false, "left": 13, "right": 14, "threshold": null}, {"id": 7, "idx": 0, "leaf": true}, {"id": 8, "idx": 1, "leaf": true}, {"id": 9, "idx": 2, "leaf": true}, {"id": 10, "idx": 3, "leaf": true}, {"id": 11, "idx": 4, "leaf": true}, {"id": 12, "idx": 5, "leaf": true}, {"id": 13, "idx": 6, "leaf": true}, {"id": 14, "idx": 7, "leaf": true}], "weights": [30.54969596862793, 0.0, 34.94964599609375, 0.0, 22.299779891967773, 24.499753952026367, 17.199825286865234, 0.0]}]}'),
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [23.72778,  23.72778,  30.408333, 35.144444, 35.144444, 30.408333, 23.72778,
                                                                                               23.72778,  18.991667, 18.991667], json.loads(
    '{"configuration":{"internals":{"compute_overlap":2,"double_precision":true,"hist_size":12,"seed":0,"upload_features":true,"use_hist_subtraction_trick":true},"method":1,"objective":0,"tree":{"alpha":0.0,"colsample_bylevel":1.0,"colsample_bytree":1.0,"eta":0.9999899864196777,"gamma_absolute":0.0,"gamma_relative":0.0,"initial_y":0.5,"labels_count":1,"lambda":0.0,"max_depth":2,"max_leaf_weight":0.0,"min_child_weight":2.0,"min_leaf_size":2,"scale_pos_weight":0.5},"verbose":{"booster":false,"data":false,"gpu":false}},"model":[{"nodes":[{"fid":6,"id":0,"leaf":false,"left":1,"right":2,"threshold":63.14999771118164},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[32.749671936035156,21.333120346069336]},{"nodes":[{"fid":11,"id":0,"leaf":false,"left":1,"right":2,"threshold":394.375},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[-2.841367483139038,1.8946772813796997]}]}'),
    2,
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_3(double_precision, use_hist_trick, upload_features): run_regression(3, [23.575,    21.575,    35.125,    33.15,     37.550003, 28.725002, 23.566666,
                                                                                               26,       17.366667, 17.366667],
                                                                                           json.loads(
    '{"configuration": {"internals": {"compute_overlap": 2, "double_precision": true, "hist_size": 12, "seed": 0, "upload_features": true, "use_hist_subtraction_trick": true}, "method": 1, "objective": 0, "tree": {"alpha": 0.0, "colsample_bylevel": 1.0, "colsample_bytree": 1.0, "eta": 0.9999899864196777, "gamma_absolute": 0.0, "gamma_relative": 0.0, "initial_y": 0.5, "labels_count": 1, "lambda": 0.0, "max_depth": 3, "max_leaf_weight": 0.0, "min_child_weight": 2.0, "min_leaf_size": 2, "scale_pos_weight": 0.5}, "verbose": {"booster": false, "data": false, "gpu": false}}, "model": [{"nodes": [{"fid": 6, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 63.14999771118164}, {"fid": 5, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": 7.072500228881836}, {"fid": 0, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 0.15729498863220215}, {"id": 3, "idx": 0, "leaf": true}, {"id": 4, "idx": 1, "leaf": true}, {"id": 5, "idx": 2, "leaf": true}, {"id": 6, "idx": 3, "leaf": true}], "weights": [30.54969596862793, 34.949649810791016, 23.399765014648438, 17.1998291015625]}, {"nodes": [{"fid": 0, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 0.031109999865293503}, {"fid": 0, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": 0.027300000190734863}, {"fid": 5, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 6.0920000076293945}, {"id": 3, "idx": 0, "leaf": true}, {"id": 4, "idx": 1, "leaf": true}, {"id": 5, "idx": 2, "leaf": true}, {"id": 6, "idx": 3, "leaf": true}], "weights": [-0.32470378279685974, -2.324706554412842, -0.33313798904418945, 2.100276231765747]}]}'),
    2,
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_4(double_precision, use_hist_trick, upload_features): run_regression(4, [23.5,      23.5,      35.2,      32.600002, 37,       28.825,    22.775,
                                                                                               25.7,      17.45,     17.45],
                                                                                           json.loads(
    '{"configuration": {"internals": {"compute_overlap": 2, "double_precision": true, "hist_size": 12, "seed": 0, "upload_features": true, "use_hist_subtraction_trick": true}, "method": 1, "objective": 0, "tree": {"alpha": 0.0, "colsample_bylevel": 1.0, "colsample_bytree": 1.0, "eta": 0.9999899864196777, "gamma_absolute": 0.0, "gamma_relative": 0.0, "initial_y": 0.5, "labels_count": 1, "lambda": 0.0, "max_depth": 4, "max_leaf_weight": 0.0, "min_child_weight": 2.0, "min_leaf_size": 2, "scale_pos_weight": 0.5}, "verbose": {"booster": false, "data": false, "gpu": false}}, "model": [{"nodes": [{"fid": 6, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 63.14999771118164}, {"fid": 5, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": 7.072500228881836}, {"fid": 0, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 0.15729498863220215}, {"fid": 0, "id": 3, "leaf": false, "left": 7, "right": 8, "threshold": null}, {"fid": 0, "id": 4, "leaf": false, "left": 9, "right": 10, "threshold": null}, {"fid": 0, "id": 5, "leaf": false, "left": 11, "right": 12, "threshold": 0.02858000062406063}, {"fid": 0, "id": 6, "leaf": false, "left": 13, "right": 14, "threshold": null}, {"id": 7, "idx": 0, "leaf": true}, {"id": 8, "idx": 1, "leaf": true}, {"id": 9, "idx": 2, "leaf": true}, {"id": 10, "idx": 3, "leaf": true}, {"id": 11, "idx": 4, "leaf": true}, {"id": 12, "idx": 5, "leaf": true}, {"id": 13, "idx": 6, "leaf": true}, {"id": 14, "idx": 7, "leaf": true}], "weights": [30.54969596862793, 0.0, 34.949649810791016, 0.0, 22.299776077270508, 24.499753952026367, 17.1998291015625, 0.0]}, {"nodes": [{"fid": 6, "id": 0, "leaf": false, "left": 1, "right": 2, "threshold": 56.45000076293945}, {"fid": 0, "id": 1, "leaf": false, "left": 3, "right": 4, "threshold": null}, {"fid": 11, "id": 2, "leaf": false, "left": 5, "right": 6, "threshold": 396.25}, {"fid": 0, "id": 3, "leaf": false, "left": 7, "right": 8, "threshold": null}, {"fid": 0, "id": 4, "leaf": false, "left": 9, "right": 10, "threshold": null}, {"fid": 11, "id": 5, "leaf": false, "left": 11, "right": 12, "threshold": 393.4749755859375}, {"fid": 0, "id": 6, "leaf": false, "left": 13, "right": 14, "threshold": null}, {"id": 7, "idx": 0, "leaf": true}, {"id": 8, "idx": 1, "leaf": true}, {"id": 9, "idx": 2, "leaf": true}, {"id": 10, "idx": 3, "leaf": true}, {"id": 11, "idx": 4, "leaf": true}, {"id": 12, "idx": 5, "leaf": true}, {"id": 13, "idx": 6, "leaf": true}, {"id": 14, "idx": 7, "leaf": true}], "weights": [1.5503127574920654, 0.0, 0.0, 0.0, -0.24976670742034912, -2.2247025966644287, 0.7002245187759399, 0.0]}]}'),
    2,
    double_precision=double_precision,
    use_hist_trick=use_hist_trick,
    upload_features=upload_features)


if __name__ == "__main__":
    # test_single_tree_depth_2(False, False, False)
    test_2trees_depth_4(True, True, True)
