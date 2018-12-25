#! /usr/bin/bash
# coding=utf-8

"""
    a function imitating cell2Mat
"""
import types
import numpy as np


# function to concatenate blocks in y-axis
def cell2mat(origin_array):
    result_array = _mergeline(origin_array[0])
    for i in range(1,origin_array.shape[0]):
        result_array = np.vstack((result_array, _mergeline(origin_array[i])))
    return result_array


# function to concatenate blocks in x-axis
def _mergeline(origin_line):
    if type(origin_line[0]) == np.ndarray:
        result_line = origin_line[0]
        for i in range(1, origin_line.shape[0]):
            result_line = np.hstack((result_line, origin_line[i]))
    else:
        result_line = origin_line
    return result_line
