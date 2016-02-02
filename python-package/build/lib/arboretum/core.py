# coding: utf-8
# pylint: disable=too-many-arguments, too-many-branches
"""Core Arboretum Library."""
from __future__ import absolute_import

import os
import ctypes

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
    return lib

_LIB = _load_lib()

def _call_and_throw_if_error(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret is not None:
        raise ArboretumError(ValueError(ret))

class DMatrix(object):
    def __init__(self, data, y = None, missing=0.0, silent=False,
                 feature_names=None, feature_types=None):

        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self._init_from_npy2d(data, missing)
        print data.shape
        if y is not None:
            assert data.shape[0] == len(y)
            self._init_y(y)

    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeDMatrix(self.handle))

    def _init_from_npy2d(self, mat, missing):
        """
        Initialize data from a 2-D numpy matrix.
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        data = np.array(mat.reshape(mat.size), dtype=np.float32)
        self.handle = ctypes.c_void_p()

        _call_and_throw_if_error(_LIB.ACreateFromDanseMatrix(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                ctypes.c_int(mat.shape[0]),
                                                ctypes.c_int(mat.shape[1]),
                                                ctypes.c_float(missing),
                                                ctypes.byref(self.handle)))


    def _init_y(self, y):
        data = np.array(y.reshape(self.rows), dtype=np.float32)
        _call_and_throw_if_error(_LIB.ASetY(self.handle,
                                            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))


class Garden(object):
    _objectives = {
        'reg:linear': 0,
        'reg:logistic': 1
    }
    def __init__(self, objective, data, depth, min_child_weight, colsample_bytree, eta):
        self.objective = objective
        self.depth = depth
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.data = data
        self.initial = 0.5
        self.eta = eta
        self._init = False


    def __del__(self):
        _call_and_throw_if_error(_LIB.AFreeGarden(self.handle))
        del self.data


    def train(self, y, y_hat):
        pass


    def grow_tree(self, grad=None):
        if not self._init:
            self.handle = ctypes.c_void_p()

            _call_and_throw_if_error(_LIB.AInitGarden(ctypes.c_int(Garden._objectives[self.objective]),
                                                      ctypes.c_int(self.depth),
                                                      ctypes.c_int(self.min_child_weight),
                                                      ctypes.c_float(self.colsample_bytree),
                                                      ctypes.c_float(self.eta),
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

    def predict(self, data):
        length = int(data.rows)
        preds = ctypes.POINTER(ctypes.c_float)()
        _call_and_throw_if_error(_LIB.APredict(self.handle,
                                                data.handle,
                                                ctypes.byref(preds)))


        if not isinstance(preds, ctypes.POINTER(ctypes.c_float)):
            raise RuntimeError('expected float pointer')
        res = np.ctypeslib.as_array(preds, shape=(length,))
        # res = np.empty(length, dtype=np.float32)

        # if not ctypes.memmove(res.ctypes.data, preds, length * res.strides[0]):
        #     raise RuntimeError('memmove failed')

        return res




