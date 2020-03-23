# coding: utf-8
# pylint: disable=too-many-arguments, too-many-branches
"""Core Arboretum Library."""
from __future__ import absolute_import

import os
import ctypes
from ctypes import *

import numpy as np
import scipy.sparse
import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


class ArboretumError(Exception):
    pass


def _load_lib():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, 'libarboretum.so')
    lib = ctypes.cdll.LoadLibrary(lib_path)
    lib.ACreateFromDenseMatrix.restype = ctypes.c_char_p
    lib.ASetY.restype = ctypes.c_char_p
    lib.AInitGarden.restype = ctypes.c_char_p
    lib.AGrowTree.restype = ctypes.c_char_p
    lib.APredict.restype = ctypes.c_char_p
    lib.AFreeDMatrix.restype = ctypes.c_char_p
    lib.AFreeGarden.restype = ctypes.c_char_p
    lib.AAppendLastTree.restype = ctypes.c_char_p
    lib.AGetY.restype = ctypes.c_char_p
    lib.ADeleteArray.restype = ctypes.c_char_p
    lib.ASetLabel.restype = ctypes.c_char_p
    lib.ASetWeights.restype = ctypes.c_char_p
    lib.ADumpModel.restype = ctypes.c_char_p
    lib.ADumpModel.argtypes = [POINTER(c_char_p), c_void_p]
    lib.ALoadModel.restype = ctypes.c_char_p
    return lib


_LIB = _load_lib()


def _call_and_throw_if_error(ret):
    if ret is not None:
        raise ArboretumError(ValueError(ret))


class DMatrix(object):
    def __init__(self, data, data_category=None, y=None, labels=None, weights=None, missing=0.0):

        self.labels_count = 1
        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self._init_from_npy2d(data, missing, category=data_category)

        if y is not None and labels is not None:
            raise ValueError(
                'y and labels both are not None. Specify labels only for multi label classification')
        if y is not None:
            assert data.shape[0] == len(y)
            self._init_y(y)
        elif labels is not None:
            self.labels_count = np.max(labels) + 1
            assert data.shape[0] == len(labels)
            self._init_labels(labels)

        if weights is not None:
            assert weights.shape[0] == self.rows
            assert weights.size == self.rows
            self._set_weight(weights)

    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeDMatrix(self.handle))

    def _set_weight(self, weights):
        data = np.array(weights.reshape(self.rows), dtype=np.float32)
        _call_and_throw_if_error(_LIB.ASetWeights(self.handle,
                                                  data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))

    def _init_from_npy2d(self, mat, missing, category=None):
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')
        if category is not None and category.dtype not in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int]:
            raise ValueError('Categoty''s type must be int like')

        data = np.array(mat.reshape(mat.size), dtype=np.float32)
        self.handle = ctypes.c_void_p()
        if category is None:
            data_category = None
            columns = 0
        else:
            columns = category.shape[1]
            data_category = np.array(category.reshape(
                category.size), dtype=np.uint32)

        _call_and_throw_if_error(_LIB.ACreateFromDenseMatrix(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                             None if data_category is None else data_category.ctypes.data_as(
                                                                 ctypes.POINTER(ctypes.c_uint)),
                                                             ctypes.c_int(
                                                                 mat.shape[0]),
                                                             ctypes.c_int(
                                                                 mat.shape[1]),
                                                             ctypes.c_int(
                                                                 columns),
                                                             ctypes.c_float(
                                                                 missing),
                                                             ctypes.byref(self.handle)))

    def _init_y(self, y):
        data = np.array(y.reshape(self.rows), dtype=np.float32)
        _call_and_throw_if_error(_LIB.ASetY(self.handle,
                                            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))

    def _init_labels(self, labels):
        data = np.array(labels.reshape(self.rows), dtype=np.uint8)
        _call_and_throw_if_error(_LIB.ASetLabel(self.handle,
                                                data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))))


class Garden(object):
    """Low level object to work with arboretum
    """

    def __init__(self, config, data=None):
        """Initialize arboretum

        Parameters
        ----------
        config : str
            Configuration as a json.
        data : DMatrix, optional
            Data used for training, by default None
        """
        self.config = config
        self.data = data
        self._init = False
        if 'labels_count' in config['tree']:
            self.labels_count = config['tree']['labels_count']
        else:
            self.labels_count = self.labels_count = 1

        self.config_str = json.dumps(config)

    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeGarden(self.handle))
        if hasattr(self, 'data'):
            del self.data

    def load(self, json_model_str):
        """Load model from json

        Parameters
        ----------
        json_model_str : str
            Json representation
        """
        json_model = json.loads(json_model_str)
        self.handle = ctypes.c_void_p()

        _call_and_throw_if_error(_LIB.AInitGarden(ctypes.c_char_p(self.config_str.encode('UTF-8')),
                                                  ctypes.byref(self.handle)))
        self._init = True

        _call_and_throw_if_error(_LIB.ALoadModel(
            c_char_p(json_model_str.encode('UTF-8')), self.handle))

    def grow_tree(self, grad=None):
        """Grows single tree

        Parameters
        ----------
        grad : numpy array, optional
            Gradient(not supported yet), by default None
        """
        if not self._init:
            self.handle = ctypes.c_void_p()

            _call_and_throw_if_error(_LIB.AInitGarden(ctypes.c_char_p(self.config_str.encode('UTF-8')),
                                                      ctypes.byref(self.handle)))
            self._init = True

        if grad:
            assert len(grad) == self.data.rows
            data = np.array(grad.reshape(self.data.rows), dtype=np.float32)
            _call_and_throw_if_error(_LIB.AGrowTree(self.handle,
                                                    self.data.handle,
                                                    data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        else:
            _call_and_throw_if_error(_LIB.AGrowTree(self.handle,
                                                    self.data.handle,
                                                    ctypes.c_void_p(grad)))

    def append_last_tree(self, data):
        """Appends last tree for ``data`` and updated prediction stored in Y.

        Parameters
        ----------
        data : DMatrix
            Data to be used to propagate through the last tree.
        """
        _call_and_throw_if_error(_LIB.AAppendLastTree(self.handle,
                                                      data.handle))

    def get_y(self, data):
        """Return prediction Y previously computed with calling ``append_last_tree`` multiple times.

        Parameters
        ----------
        data : DMatrix
            data input

        Returns
        -------
        numpy array
            y

        Raises
        ------
        RuntimeError
            [description]
        """
        length = int(data.rows)
        preds = ctypes.POINTER(ctypes.c_float)()
        _call_and_throw_if_error(_LIB.AGetY(self.handle,
                                            data.handle,
                                            ctypes.byref(preds)))

        if not isinstance(preds, ctypes.POINTER(ctypes.c_float)):
            raise RuntimeError('expected float pointer')

        if self.labels_count == 1:
            res = np.copy(np.ctypeslib.as_array(preds, shape=(length,)))
        else:
            res = np.copy(np.ctypeslib.as_array(
                preds, shape=(length, self.labels_count)))

        _call_and_throw_if_error(_LIB.ADeleteArray(preds))

        return res

    def predict(self, data, n_rounds=-1):
        """Predict

        Parameters
        ----------
        data : DMatrix
            Data input
        n_rounds : int, optional
            [description], by default -1

        Returns
        -------
        numpy array
            prediction

        Raises
        ------
        RuntimeError
            [description]
        """
        length = int(data.rows)
        preds = ctypes.POINTER(ctypes.c_float)()
        _call_and_throw_if_error(_LIB.APredict(self.handle,
                                               data.handle,
                                               ctypes.byref(preds), n_rounds))

        if not isinstance(preds, ctypes.POINTER(ctypes.c_float)):
            raise RuntimeError('expected float pointer')

        if self.labels_count == 1:
            res = np.copy(np.ctypeslib.as_array(preds, shape=(length,)))
        else:
            res = np.copy(np.ctypeslib.as_array(
                preds, shape=(length, self.labels_count)))

        _call_and_throw_if_error(_LIB.ADeleteArray(preds))

        return res

    def dump(self):
        """Dumps the model as a json

        Returns
        -------
        str
            json
        """
        json_p = c_char_p()
        _call_and_throw_if_error(_LIB.ADumpModel(
            ctypes.byref(json_p), self.handle))
        return json_p.value.decode('utf-8')


def train(config, data, num_round):
    """Train model according to the parameters

    Parameters
    ----------
    config : str
        configuration as a json
    data : DMatrix
        Data to be trained on.
    num_round : int
        Number of boosting rounds

    Returns
    -------
    Garden
        The trained model.
    """
    model = Garden(config)
    model.data = data
    model.labels_count = data.labels_count
    for _ in range(num_round):
        model.grow_tree(None)
    return model


def load(json_model_str):
    """load model from json

    Parameters
    ----------
    json_model_str : str
        json model representation

    Returns
    -------
    self : object
        Returns self.
    """
    json_model = json.loads(json_model_str)
    config = json_model['configuration']
    model = Garden(config)
    model.load(json_model_str)
    return model


class ArboretumRegression(object):
    """Scikit-learn API like implementation for regression.

    """

    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100,
                 verbosity=1,
                 gamma_absolute=0.0, gamma_relative=0.0,
                 min_child_weight=1.0, min_leaf_size=0,  max_leaf_weight=0.0, colsample_bytree=0.8,
                 colsample_bylevel=1.0, l1=1.0, l2=1.0,
                 scale_pos_weight=1.0, initial_y=0.5, seed=0,
                 double_precision=False, method='hist', hist_size=255, **kwargs):
        """[summary]

        Parameters
        ----------
        max_depth : int, optional
            Maximum tree depth, by default 6
        learning_rate : float, optional
            Learning rate, by default 0.1
        n_estimators : int, optional
            Number of boosted trees to fit, by default 100
        verbosity : int, optional
            verbosity, by default 1
        gamma_absolute : float, optional
            Minimum absolute gain required to make a further partition on a leaf, by default 0.0
        gamma_relative : float, optional
            Minimum relative(split vs constant) gain required to make a further partition on a leaf, by default 0.0, by default 0.0
        min_child_weight : float, optional
            Minimum sum of hessing to allow split, by default 1.0
        min_leaf_size : int, optional
            Minimum number of samples in a leaf, by default 0
        max_leaf_weight : float, optional
            Maximum weight of a leaf (values less than ``-max_leaf_weight`` and greater than ``max_leaf_weight``
            will be tranceted to ``max_leaf_weight`` and ``max_leaf_weight`` respectively). Zero value is ignored, by default 0.0
        colsample_bytree : float, optional
            Subsample ratio of columns when constructing each tree., by default 0.8
        colsample_bylevel : float, optional
            Subsample ratio of columns when constructing each tree's level., by default 1.0
        l1 : float, optional
            L1 or alpha regularization, by default 1.0
        l2 : float, optional
            L2 or lambda regularization, by default 1.0
        scale_pos_weight : float, optional
            Scaling ratio for positive , by default 1.0
        initial_y : float, optional
            Initial value to start from, by default 0.5
        seed : int, optional
            Seed for random number generator., by default 0
        double_precision : bool, optional
            Use double precision to summation. Makes result run-to-run reproducible, but reduces performance a bit(~10%)., by default False
        method : str, optional
            Algorithm to grow trees. 'exact' or 'hist'., by default 'hist'
        hist_size : int, optional
            Histogram size, only used by when ``method`` is 'hist', by default 255
        """
        config = {'objective': 0,
                  'method': 1 if method == 'hist' else 0,
                  'internals':
                  {
                      'double_precision': double_precision,
                      'compute_overlap': 2,
                      'use_hist_subtraction_trick': True,
                      'dynamic_parallelism': True,
                      'upload_features': True,
                      'hist_size': hist_size,
                      'seed': seed,
                  },
                  'verbose':
                  {
                      'gpu': True if verbosity > 0 else False,
                      'booster': True if verbosity > 0 else False,
                      'data': True if verbosity > 0 else False,
                  },
                  'tree':
                  {
                      'eta': learning_rate,
                      'max_depth': max_depth,
                      'gamma_absolute': gamma_absolute,
                      'gamma_relative': gamma_relative,
                      'min_child_weight': min_child_weight,
                      'min_leaf_size': min_leaf_size,
                      'colsample_bytree': colsample_bytree,
                      'colsample_bylevel': colsample_bylevel,
                      'max_leaf_weight': max_leaf_weight,
                      'lambda': l2,
                      'alpha': l1
                  }}
        self._config = config
        self.n_estimators = n_estimators
        self._garden = Garden(self._config)
        self.verbosity = verbosity

    def fit(self, X, y=None, eval_set=None, eval_labels=None, early_stopping_rounds=5,
            eval_metric=mean_squared_error):
        """Fit gradient boosting model.

        Parameters
        ----------
        X : DMatrix or numpy array
            Data to fit
        y : numpy array, optional
            labels, by default None
        eval_set : DMatrix or numpy_array, optional
            Evaluation set data used for early stopping., by default None
        eval_labels : numpy array, optional
            Evaluation set labels, by default None
        early_stopping_rounds : int, optional
            Stop fitting process if there's no improvement for ``eval_set`` during
            ``early_stopping_rounds`` rounds., by default 5

        Returns
        -------
        self
            [description]

        Raises
        ------
        ArgumentError
            [description]
        ArgumentError
            [description]
        """
        data = None
        if isinstance(X, DMatrix):
            data = X
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            data = DMatrix(X, y=y)
        else:
            raise ArgumentError("Only DMatrix and numpy array are supported")

        self._garden.data = data

        eval_data = None
        if eval_set is not None:
            if isinstance(eval_set, DMatrix):
                eval_data = X
            elif isinstance(eval_set, np.ndarray):
                eval_data = DMatrix(eval_set)
            else:
                raise ArgumentError(
                    "Only DMatrix and numpy array are supported")

        self.best_round = -1
        self.best_score = np.inf

        for i in range(self.n_estimators):
            self._garden.grow_tree()
            if eval_data is not None:

                self._garden.append_last_tree(eval_data)
                pred = self._garden.get_y(eval_data)
                score = eval_metric(eval_labels, pred)

                if score < self.best_score:
                    print(
                        "improved score {0} {1}->{2}".format(i, self.best_score, score))
                    self.best_score = score
                    self.best_round = i

                if early_stopping_rounds + self.best_round < i:
                    print("early stopping at {0} score {1}, use 'best_round' and 'best_score' to get it".format(
                        self.best_round, self.best_score))
                    break

        return self

    def predict(self, X, n_rounds=-1):
        """Predict with ``X``.

        Parameters
        ----------
        X : DMatrix or numpy array
            Data
        n_rounds : int, optional
            Number of trees to use, -1 - use all, by default -1

        Returns
        -------
        numpy array
            Prediction

        Raises
        ------
        ArgumentError
            [description]
        """
        data = None
        if isinstance(X, DMatrix):
            data = X
        elif isinstance(X, np.ndarray):
            data = DMatrix(X)
        else:
            raise ArgumentError("Only DMatrix and numpy array are supported")
        return self._garden.predict(data, n_rounds)


class ArboretumClassifier(object):
    """Scikit-learn API like implementation for regression.

    """

    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100,
                 verbosity=1,
                 gamma_absolute=0.0, gamma_relative=0.0,
                 min_child_weight=1.0, min_leaf_size=0,  max_leaf_weight=0.0, colsample_bytree=0.8,
                 colsample_bylevel=1.0, l1=1.0, l2=1.0,
                 scale_pos_weight=1.0, initial_y=0.5, seed=0,
                 double_precision=False, method='hist', hist_size=255, **kwargs):
        """[summary]

        Parameters
        ----------
        max_depth : int, optional
            Maximum tree depth, by default 6
        learning_rate : float, optional
            Learning rate, by default 0.1
        n_estimators : int, optional
            Number of boosted trees to fit, by default 100
        verbosity : int, optional
            verbosity, by default 1
        gamma_absolute : float, optional
            Minimum absolute gain required to make a further partition on a leaf, by default 0.0
        gamma_relative : float, optional
            Minimum relative(split vs constant) gain required to make a further partition on a leaf, by default 0.0, by default 0.0
        min_child_weight : float, optional
            Minimum sum of hessing to allow split, by default 1.0
        min_leaf_size : int, optional
            Minimum number of samples in a leaf, by default 0
        max_leaf_weight : float, optional
            Maximum weight of a leaf (values less than ``-max_leaf_weight`` and greater than ``max_leaf_weight``
            will be tranceted to ``max_leaf_weight`` and ``max_leaf_weight`` respectively). Zero value is ignored, by default 0.0
        colsample_bytree : float, optional
            Subsample ratio of columns when constructing each tree., by default 0.8
        colsample_bylevel : float, optional
            Subsample ratio of columns when constructing each tree's level., by default 1.0
        l1 : float, optional
            L1 or alpha regularization, by default 1.0
        l2 : float, optional
            L2 or lambda regularization, by default 1.0
        scale_pos_weight : float, optional
            Scaling ratio for positive , by default 1.0
        initial_y : float, optional
            Initial value to start from, by default 0.5
        seed : int, optional
            Seed for random number generator., by default 0
        double_precision : bool, optional
            Use double precision to summation. Makes result run-to-run reproducible, but reduces performance a bit(~10%)., by default False
        method : str, optional
            Algorithm to grow trees. 'exact' or 'hist'., by default 'hist'
        hist_size : int, optional
            Histogram size, only used by when ``method`` is 'hist', by default 255
        """
        config = {'objective': 1,
                  'method': 1 if method == 'hist' else 0,
                  'internals':
                  {
                      'double_precision': double_precision,
                      'compute_overlap': 2,
                      'use_hist_subtraction_trick': True,
                      'dynamic_parallelism': True,
                      'upload_features': True,
                      'hist_size': hist_size,
                      'seed': seed,
                  },
                  'verbose':
                  {
                      'gpu': True if verbosity > 0 else False,
                      'booster': True if verbosity > 0 else False,
                      'data': True if verbosity > 0 else False,
                  },
                  'tree':
                  {
                      'eta': learning_rate,
                      'max_depth': max_depth,
                      'gamma_absolute': gamma_absolute,
                      'gamma_relative': gamma_relative,
                      'min_child_weight': min_child_weight,
                      'min_leaf_size': min_leaf_size,
                      'colsample_bytree': colsample_bytree,
                      'colsample_bylevel': colsample_bylevel,
                      'max_leaf_weight': max_leaf_weight,
                      'lambda': l2,
                      'alpha': l1
                  }}
        self._config = config
        self.n_estimators = n_estimators
        self._garden = Garden(self._config)
        self.verbosity = verbosity

    def fit(self, X, y=None, eval_set=None, eval_labels=None, early_stopping_rounds=5, eval_metric=lambda a, b: -roc_auc_score(a, b)):
        """Fit gradient boosting model.

        Parameters
        ----------
        X : DMatrix or numpy array
            Data to fit
        y : numpy array, optional
            labels, by default None
        eval_set : DMatrix or numpy_array, optional
            Evaluation set data used for early stopping., by default None
        eval_labels : numpy array, optional
            Evaluation set labels, by default None
        early_stopping_rounds : int, optional
            Stop fitting process if there's no improvement for ``eval_set`` during
            ``early_stopping_rounds`` rounds., by default 5

        Returns
        -------
        self
            [description]

        Raises
        ------
        ArgumentError
            [description]
        ArgumentError
            [description]
        """
        data = None
        if isinstance(X, DMatrix):
            data = X
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            data = DMatrix(X, y=y)
        else:
            raise ArgumentError("Only DMatrix and numpy array are supported")

        self._garden.data = data

        eval_data = None
        if eval_set is not None:
            if isinstance(eval_set, DMatrix):
                eval_data = X
            elif isinstance(eval_set, np.ndarray):
                eval_data = DMatrix(eval_set)
            else:
                raise ArgumentError(
                    "Only DMatrix and numpy array are supported")

        self.best_round = -1
        self.best_score = np.inf

        for i in range(self.n_estimators):
            self._garden.grow_tree()
            if eval_data is not None:
                from sklearn.metrics import mean_squared_error
                self._garden.append_last_tree(eval_data)
                pred = self._garden.get_y(eval_data)
                score = eval_metric(eval_labels, pred)

                if score < self.best_score:
                    print(
                        "improved score {0} {1}->{2}".format(i, self.best_score, score))
                    self.best_score = score
                    self.best_round = i

                if early_stopping_rounds + self.best_round < i:
                    print("early stopping at {0} score {1}, use 'best_round' and 'best_score' to get it".format(
                        self.best_round, self.best_score))
                    break

        return self

    def predict(self, X, n_rounds=-1):
        """Predict with ``X``.

        Parameters
        ----------
        X : DMatrix or numpy array
            Data
        n_rounds : int, optional
            Number of trees to use, -1 - use all, by default -1

        Returns
        -------
        numpy array
            Positive class probability

        Raises
        ------
        ArgumentError
            [description]
        """
        data = None
        if isinstance(X, DMatrix):
            data = X
        elif isinstance(X, np.ndarray):
            data = DMatrix(X)
        else:
            raise ArgumentError("Only DMatrix and numpy array are supported")
        return self._garden.predict(data, n_rounds)
