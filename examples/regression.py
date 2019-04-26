import arboretum
import numpy as np
from sklearn.datasets import load_boston
import json
from sklearn.model_selection import train_test_split


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.mean(diff))


if __name__ == "__main__":
    boston = load_boston()

    data_train, data_test, y_train,  y_test = train_test_split(
        boston.data, boston.target, test_size=0.2, random_state=42)

    data = arboretum.DMatrix(data_train, y=y_train)

    config = json.dumps({'objective': 0,
                         'internals':
                         {
                             'double_precision': True
                         },
                         'verbose':
                         {
                             'gpu': True
                         },
                         'tree':
                         {
                             'eta': 1.0,
                             'max_depth': 5,
                             'gamma': 0.0,
                             'min_child_weight': 2,
                             'min_leaf_size': 2,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0
                         }})

    # init model
    model = arboretum.Garden(config, data)

    # grow trees
    for i in range(200):
        model.grow_tree()

    # predict on train data set
    pred_train = model.predict(data)

    pred_test = model.predict(arboretum.DMatrix(data_test))

    # print first n records from train set
    print(pred_train[0:10], y_train[0:10])

    # print first n records from test set
    print(pred_test[0:10], y_test[0:10])

    # RMSE
    print('train:', rmse(pred_train, y_train),
          'test:', rmse(pred_test, y_test))
