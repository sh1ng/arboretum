import arboretum
import numpy as np
from sklearn.datasets import load_boston
import xgboost
import json


def rmse(y, y_hat):
    diff = np.power(y - y_hat, 2)
    return np.sqrt(np.sum(diff))

# load test data
boston = load_boston()
n = 10

# create data matrix
data = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
y = boston.target[0:n]

config = json.dumps({'objective':0,
                     'internals':
                             {
                                 'double_precision':True
                             },
                     'verbose':
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

# predict on train data set
pred = model.predict(data)

# print first n records
print(pred[0:10])

#RMSE
print(rmse(pred, y))


# xgboost as refernce value
data1 = arboretum.DMatrix(boston.data[0:n], y=boston.target[0:n])
y1 = boston.target[0:n]

pred = model.predict(data1)

print(pred[0:10])
print(rmse(pred, y1))

mat = xgboost.DMatrix(boston.data[0:n], label=boston.target[0:n])
param = {'max_depth':5,
         'silent':True, 'objective':'reg:linear' }
param['nthread'] = 1
param['min_child_weight'] = 2
param['colspan_by_tree'] = 1.0
param['colspan_by_level'] = 1.0
param['eval_metric'] = 'rmse'
param['lambda'] = 0.0
param['eta'] = 1.0
param['gamma'] = 0.0
param['alpha'] = 0.0

model = xgboost.train(param, mat, 5)
pred_xgb = model.predict(mat)
print(pred_xgb[0:10])
print(boston.target[0:10])

print(rmse(pred_xgb, y))

print(np.count_nonzero(pred != pred_xgb))
assert np.count_nonzero(pred != pred_xgb) == 0
