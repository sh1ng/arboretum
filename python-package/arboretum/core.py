# coding: utf-8
# pylint: disable=too-many-arguments, too-many-branches
"""Core Arboretum Library."""
from __future__ import absolute_import

import os
import ctypes
from ctypes import *

import numpy as np
import scipy.sparse

class ArboretumError(Exception):
    pass

def _load_lib():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, 'arboretum_wrapper.so')
    lib = ctypes.cdll.LoadLibrary(lib_path)
    lib.ACreateFromDanseMatrix.restype = ctypes.c_char_p
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
    return lib

_LIB = _load_lib()

def _call_and_throw_if_error(ret):
    if ret is not None:
        raise ArboretumError(ValueError(ret))

class DMatrix(object):
    def __init__(self, data, data_category = None, y=None, labels=None,  missing=0.0):

        self.labels_count = 1
        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self._init_from_npy2d(data, missing, category = data_category)

        if y is not None and labels is not None:
            raise ValueError('y and labels both are not None. Specify labels only for multi label classification')
        if y is not None:
            assert data.shape[0] == len(y)
            self._init_y(y)
        elif labels is not None:
            self.labels_count = np.max(labels) + 1
            assert data.shape[0] == len(labels)
            self._init_labels(labels)

    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeDMatrix(self.handle))

    def _init_from_npy2d(self, mat, missing, category = None):
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
            data_category = np.array(category.reshape(category.size), dtype=np.uint32)

        _call_and_throw_if_error(_LIB.ACreateFromDanseMatrix(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                             None if data_category is None else data_category.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                                ctypes.c_int(mat.shape[0]),
                                                ctypes.c_int(mat.shape[1]),
                                                ctypes.c_int(columns),
                                                ctypes.c_float(missing),
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
    _objectives = {
        'reg:linear': 0,
        'reg:logistic': 1
    }
    def __init__(self, config, data):
        self.data = data
        self._config = config
        self._init = False
        self.labels_count = data.labels_count



    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeGarden(self.handle))
        del self.data


    def train(self, y, y_hat):
        pass


    def grow_tree(self, grad=None):
        if not self._init:
            self.handle = ctypes.c_void_p()

            _call_and_throw_if_error(_LIB.AInitGarden(ctypes.c_char_p(self._config.encode('UTF-8')),
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
            res = np.copy(np.ctypeslib.as_array(preds, shape=(length, self.labels_count)))


        _call_and_throw_if_error(_LIB.ADeleteArray(preds))

        return res

    def predict(self, data):
        length = int(data.rows)
        preds = ctypes.POINTER(ctypes.c_float)()
        _call_and_throw_if_error(_LIB.APredict(self.handle,
                                                data.handle,
                                                ctypes.byref(preds)))


        if not isinstance(preds, ctypes.POINTER(ctypes.c_float)):
            raise RuntimeError('expected float pointer')

        if self.labels_count == 1:
            res = np.copy(np.ctypeslib.as_array(preds, shape=(length,)))
        else:
            res = np.copy(np.ctypeslib.as_array(preds, shape=(length, self.labels_count)))

        _call_and_throw_if_error(_LIB.ADeleteArray(preds))

        return res




