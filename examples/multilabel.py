import arboretum
import numpy as np
from sklearn.datasets import load_iris
import xgboost
import json


# load test data
iris = load_iris()
n = 10000

y = iris.target[0:n]
# create data matrix
data = arboretum.DMatrix(iris.data[:, 0:n], labels = y)


config = json.dumps({'objective':3, 'verbose':
{
'gpu': True,
    'booster': True
},
'tree':
{
    'labels_count':3,
'eta': 0.2,
'max_depth': 6,
'gamma': 0.0,
'min_child_weight': 2.0,
'min_leaf_size': 0,
'colsample_bytree': 1.0,
'colsample_bylevel': 1.0,
'lambda': 0.0,
'alpha': 0.0,
'initial_y':0.0
}})

model = arboretum.Garden(config, data)

# grow trees
for i in range(10):
    model.grow_tree()

# pred = model.get_y(data)
#
# print(pred)

# predict on train data set
pred = model.predict(data)

# print first n records
print(pred)

mat = xgboost.DMatrix(iris.data[0:n], label=y)
param = {'max_depth':5,
         'silent':True, 'objective': 'multi:softprob' }
param['nthread'] = 1
param['min_child_weight'] = 2
param['colspan_by_tree'] = 1.0
param['colspan_by_level'] = 1.0
param['eval_metric'] = 'rmse'
param['lambda'] = 0.0
param['eta'] = 0.2
param['gamma'] = 0.0
param['alpha'] = 0.0
param['num_class'] = 3

model = xgboost.train(param, mat, 10)
pred = model.predict(mat)
print(pred)
print(y)

