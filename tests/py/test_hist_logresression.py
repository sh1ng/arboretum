import arboretum
import numpy as np
from sklearn.datasets import load_iris
import json
import pytest
import utils


def run_regression(depth, true_values, true_model, trees=1, double_precision=True, use_hist_trick=True, upload_features=True):
        # load test data
    iris = load_iris()
    n = 10000

    y = iris.target != 2

    print(iris.data.shape)

    data = arboretum.DMatrix(iris.data, y=y)

    config = json.dumps({'objective': 1,
                         'method': 1,
                         'internals':
                         {
                             'double_precision': double_precision,
                             'compute_overlap': 2,
                             'use_hist_subtraction_trick': use_hist_trick,
                             'upload_features': upload_features,
                         },
                         'tree':
                         {
                             'eta': 0.2,
                             'max_depth': depth,
                             'gamma': 0.0,
                             'min_child_weight': 2.0,
                             'min_leaf_size': 0,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0
                         }})

    model = arboretum.Garden(config, data)

    # print(model.handle)

    # grow trees
    for i in range(trees):
        model.grow_tree()

    print(model.handle)
    # predict on train data set
    pred = model.predict(data)
    print(true_values)
    print(pred)
    pred = model.predict(data)
    assert np.allclose(pred, true_values)
    model_json = model.dump()
    model = json.loads(model_json)
    utils.assert_model(true_model, model)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.40549785, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.5894128,
                                                                                                    0.5894128, 0.5894128, 0.5894128, 0.5894128, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.5894128, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.5894128,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.5894128, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.5894128, 0.5894128, 0.40549785, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785,
                                                                                                    0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785, 0.40549785],
                                                                                                json.loads(
                                                                                                    '{"configuration":{"internals":{"compute_overlap":2,"double_precision":false,"hist_size":255,"seed":0,"upload_features":true,"use_hist_subtraction_trick":true},"method":1,"objective":1,"tree":{"alpha":0.0,"colsample_bylevel":1.0,"colsample_bytree":1.0,"eta":0.20000000298023224,"gamma_absolute":0.0,"gamma_relative":0.0,"initial_y":0.5,"labels_count":1,"lambda":0.0,"max_depth":2,"max_leaf_weight":0.0,"min_child_weight":2.0,"min_leaf_size":0,"scale_pos_weight":0.5},"verbose":{"booster":false,"data":false,"gpu":false}},"model":[{"nodes":[{"fid":3,"id":0,"leaf":false,"left":1,"right":2,"threshold":1.75},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[0.36153846979141235,-0.38260871171951294]}]}'),
                                                                                                double_precision=double_precision,
                                                                                                use_hist_trick=use_hist_trick,
                                                                                                upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_single_tree_depth_3(double_precision, use_hist_trick, upload_features): run_regression(3, [0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386, 0.59668386,
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
                                                                                                    0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.40131232, 0.42555746],
                                                                                                json.loads(
                                                                                                    '{"configuration":{"internals":{"compute_overlap":2,"double_precision":false,"hist_size":255,"seed":0,"upload_features":true,"use_hist_subtraction_trick":true},"method":1,"objective":1,"tree":{"alpha":0.0,"colsample_bylevel":1.0,"colsample_bytree":1.0,"eta":0.20000000298023224,"gamma_absolute":0.0,"gamma_relative":0.0,"initial_y":0.5,"labels_count":1,"lambda":0.0,"max_depth":3,"max_leaf_weight":0.0,"min_child_weight":2.0,"min_leaf_size":0,"scale_pos_weight":0.5},"verbose":{"booster":false,"data":false,"gpu":false}},"model":[{"nodes":[{"fid":3,"id":0,"leaf":false,"left":1,"right":2,"threshold":1.75},{"fid":2,"id":1,"leaf":false,"left":3,"right":4,"threshold":4.850000381469727},{"fid":0,"id":2,"leaf":false,"left":5,"right":6,"threshold":6.050000190734863},{"id":3,"idx":0,"leaf":true},{"id":4,"idx":1,"leaf":true},{"id":5,"idx":2,"leaf":true},{"id":6,"idx":3,"leaf":true}],"weights":[0.3916666805744171,0.0,-0.30000001192092896,-0.4000000059604645]}]}'),
                                                                                                double_precision=double_precision,
                                                                                                use_hist_trick=use_hist_trick,
                                                                                                upload_features=upload_features)


@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("use_hist_trick", [True, False])
@pytest.mark.parametrize("upload_features", [True, False])
def test_2trees_depth_2(double_precision, use_hist_trick, upload_features): run_regression(2, [0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.5166032, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.48408338, 0.6638412,
                                                                                               0.5166032, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.5166032,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.5166032,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.6638412,
                                                                                               0.6638412, 0.6638412, 0.6638412, 0.6638412, 0.3367726, 0.3367726,
                                                                                               0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.6638412, 0.3367726,
                                                                                               0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726,
                                                                                               0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.5166032,
                                                                                               0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726,
                                                                                               0.48408338, 0.3367726, 0.3367726, 0.5166032, 0.3367726, 0.3367726,
                                                                                               0.3367726, 0.5166032, 0.5166032, 0.3367726, 0.3367726, 0.3367726,
                                                                                               0.48408338, 0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726,
                                                                                               0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726, 0.3367726],
                                                                                           json.loads(
                                                                                               '{"configuration":{"internals":{"compute_overlap":2,"double_precision":false,"hist_size":255,"seed":0,"upload_features":true,"use_hist_subtraction_trick":true},"method":1,"objective":1,"tree":{"alpha":0.0,"colsample_bylevel":1.0,"colsample_bytree":1.0,"eta":0.20000000298023224,"gamma_absolute":0.0,"gamma_relative":0.0,"initial_y":0.5,"labels_count":1,"lambda":0.0,"max_depth":2,"max_leaf_weight":0.0,"min_child_weight":2.0,"min_leaf_size":0,"scale_pos_weight":0.5},"verbose":{"booster":false,"data":false,"gpu":false}},"model":[{"nodes":[{"fid":3,"id":0,"leaf":false,"left":1,"right":2,"threshold":1.75},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[0.36153846979141235,-0.38260871171951294]},{"nodes":[{"fid":2,"id":0,"leaf":false,"left":1,"right":2,"threshold":4.850000381469727},{"id":1,"idx":0,"leaf":true},{"id":2,"idx":1,"leaf":true}],"weights":[0.318920761346817,-0.29510125517845154]}]}'),
                                                                                           2,
                                                                                           double_precision=double_precision,
                                                                                           use_hist_trick=use_hist_trick,
                                                                                           upload_features=upload_features)


if __name__ == "__main__":
    test_2trees_depth_2(False, True, True)
