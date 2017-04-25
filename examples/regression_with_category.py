import arboretum
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import json
from scipy.sparse import csc_matrix

def convert2category(x):
    unq = np.unique(x)
    return np.searchsorted(unq, x).astype(np.uint32)


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.sum(diff))

# load test data
boston = load_boston()
n = 4000

categoties = [3]
data_categories = []
for item in categoties:
    data_categories.append(convert2category(boston.data[:, item]))

data_categories = np.stack(data_categories, axis=-1)

data_source = boston.data[:,4:5]

# create data matrix
data = arboretum.DMatrix(data_source[0:n], data_category=data_categories, y=boston.target[0:n])
y = boston.target[0:n]

config = json.dumps({'objective':0, 
'internals':
{
'double_precision': True,
'compute_overlap': 2 
},
'verbose':
{
'gpu': True
},
'tree':
{
'eta': 0.5,
'max_depth': 10,
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
for i in range(10):
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

print(y[0:10])


print('-'*30)

data_source = boston.data[:,3:5]

# create data matrix
data = arboretum.DMatrix(data_source[0:n], y=boston.target[0:n])
y = boston.target[0:n]

config = json.dumps({'objective':0, 
'internals':
{
'double_precision': True,
'compute_overlap': 2
},
'verbose':
{
'gpu': True
},
'tree':
{
'eta': 0.5,
'max_depth': 10,
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
for i in range(10):
    model.grow_tree()

pred1 = model.get_y(data)
# print first n records
print(pred1[0:10])

#RMSE
print(rmse(pred1, y))

diff = pred != pred1
print(np.count_nonzero(pred != pred1))
if np.count_nonzero(pred != pred1) > 0:
    print(pred1[diff], pred[diff], y[diff])


# predict on train data set
pred1 = model.predict(data)

# print first n records
print(pred1[0:10])

#RMSE
print(rmse(pred1, y))

print(y[0:10])

diff = pred != pred1
print(np.count_nonzero(pred != pred1))
if np.count_nonzero(pred != pred1) > 0:
    print(pred1[diff], pred[diff], y[diff])


