import arboretum
import numpy as np
from sklearn.datasets import load_iris
import json


if __name__ == "__main__":
    # load test data
    iris = load_iris()
    n = 10000

    y = iris.target[0:n]
    # create data matrix
    data = arboretum.DMatrix(iris.data[:, 0:n], labels=y)

    config = json.dumps({'objective': 3, 'verbose':
                         {
                             'gpu': True,
                             'booster': True
                         },
                         'internals':
                         {
                             'double_precision': True
                         },
                         'tree':
                         {
                             'labels_count': 3,
                             'eta': 0.2,
                             'max_depth': 4,
                             'gamma': 0.0,
                             'min_child_weight': 2.0,
                             'min_leaf_size': 0,
                             'colsample_bytree': 1.0,
                             'colsample_bylevel': 1.0,
                             'lambda': 0.0,
                             'alpha': 0.0,
                             'initial_y': 0.0
                         }})

    model = arboretum.Garden(config, data)

    # grow trees
    for i in range(2):
        model.grow_tree()

    pred = model.predict(data)

    print(pred)
