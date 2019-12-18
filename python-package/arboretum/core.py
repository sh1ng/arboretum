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
    def __init__(self, config, data=None):
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
        json_model = json.loads(json_model_str)
        self.handle = ctypes.c_void_p()

        _call_and_throw_if_error(_LIB.AInitGarden(ctypes.c_char_p(self.config_str.encode('UTF-8')),
                                                  ctypes.byref(self.handle)))
        self._init = True

        _call_and_throw_if_error(_LIB.ALoadModel(
            c_char_p(json_model_str.encode('UTF-8')), self.handle))

    def train(self, num_round):
        pass

    def grow_tree(self, grad=None):
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
        _call_and_throw_if_error(_LIB.AAppendLastTree(self.handle,
                                                      data.handle))

    def get_y(self, data):
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
        json = c_char_p()
        _call_and_throw_if_error(_LIB.ADumpModel(
            ctypes.byref(json), self.handle))
        return json.value.decode('utf-8')


def train(config, data, num_round):
    model = Garden(config)
    model.data = data
    model.labels_count = data.labels_count
    for _ in range(num_round):
        model.grow_tree(None)
    return model


def load(json_model_str):
    json_model = json.loads(json_model_str)
    config = json_model['configuration']
    model = Garden(config)
    model.load(json_model_str)
    return model
