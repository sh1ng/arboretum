# import arboretum
import numpy as np
from sklearn.datasets import load_iris
import json
import arboretum


# load test data
iris = load_iris()
n = 10000

index = iris.target != 2

y = iris.target[index][0:n]
# # create data matrix
data = arboretum.DMatrix(iris.data[index, 0:n], y=y)


config = {'objective': 1, 'verbose':
          {
              'gpu': True
          },
          'tree':
          {
              'eta': 0.2,
              'max_depth': 6,
              'gamma': 0.0,
              'min_child_weight': 2.0,
              'min_leaf_size': 0,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.8,
              'lambda': 0.0,
              'alpha': 0.0
          }}

model = arboretum.Garden(config, data)

# grow trees
for i in range(2):
    model.grow_tree()

# predict on train data set
pred = model.predict(data)

# print first n records
print(pred)
