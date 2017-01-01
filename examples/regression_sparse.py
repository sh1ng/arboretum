import arboretum
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import json
from scipy.sparse import csc_matrix


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.sum(diff))

# load test data
boston = load_boston()
n = 10000

csc = csc_matrix(pd.get_dummies(boston.data[0:n, 3]))

data = np.delete(boston.data, 3, axis=1)

# create data matrix
data = arboretum.DMatrix(data[0:n], data_csc=csc, y=boston.target[0:n])
y = boston.target[0:n]

config = json.dumps({'objective':0, 'verbose':
{
'gpu': True
},
'tree':
{
'eta': 1.0,
'max_depth': 6,
'gamma': 0.0,
'min_child_weight': 2,
'min_leaf_size': 2,
'colsample_bytree': 1.0,
'colsample_bylevel': 1.0,
'lambda': 0.0,
'alpha': 0.0
}})

# init model
#model = arboretum.Garden('reg:linear', data, 6, 2, 1, 0.5)
model = arboretum.Garden(config, data)

# grow trees
for i in range(5):
    model.grow_tree()

pred = model.get_y(data)
# print first n records
print(pred[0:10])

#RMSE
print(rmse(pred, y))


# predict on train data set
pred = model.predict(data)

# print first n records
print(pred[0:10])

#RMSE
print(rmse(pred, y))
