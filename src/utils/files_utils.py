from natsort import natsorted
import unicodedata
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def natural_string_sort(list2sort):
    return natsorted(list2sort, key=lambda y: y.lower())

# Definimos la función de CORRECCIÓN de TILDES y Ñs
def strip_accents(s):
    """
    Remove accents,ñ and other symbols
            Args:
    			s (str): String to be stripped
    		Return:
    			string withouth punctuation symbols and others
    """
    return ''.join(c for c in unicodedata.normalize('NFD', (s))if unicodedata.category(c) != 'Mn')


def get_max_index(list_array):
    """
        Get indexes of maximum values
           Args:
            list_array (list of numbers): List of integers from which we want to extract the index, as well as the maximum value selected
           Return:
           max_value (number): Maximum of the list_array
           possible_occurencies (list of int): List of indexes where the maximum appear
    """
    max_value = 0
    possible_occurencies = []
    for i in range(len(list_array)):
        val = list_array[i]
        if(val > max_value):
            max_value = val
            possible_occurencies = [i]
        elif(val==max_value):
            possible_occurencies.append(i)
        else:
            continue
    return max_value, possible_occurencies

def get_local_max(matrix):
    #Get local max. by cols
    list_max = []
    if(len(matrix.shape)>=2):
        for col in range(matrix.shape[-1]):
            new_max_index = list(argrelextrema(matrix[:,col], np.greater)[0])
            #Check first value
            if(len(new_max_index)>0 and matrix[new_max_index[0]-1,col]==-1):
                new_max_index = new_max_index[1::]
            list_max+=list(matrix[new_max_index, col])
    else:
        new_max_index = list(argrelextrema(matrix, np.greater)[0])
        if (matrix[new_max_index[0] - 1] == -1):
            new_max_index = new_max_index[1::]
        list_max+=list(matrix[new_max_index])
    # Add global max.
    list_max += [np.max(matrix)]
    return list_max

def truncate(n):
    return round(n * 100) / 100


def takeClosest(num,collection):
   return min(collection,key=lambda x:abs(x-num))

def truncateHighestClosest(num, collection):
    best_index = []
    for val in collection:
        if(val<num):
            best_index.append(1)
        else:
            best_index.append(abs(val-num))
    return collection[np.argmin(best_index)]

def truncateLowestClosest(num, collection):
    best_index = []
    for val in collection:
        if(val>num):
            best_index.append(1)
        else:
            best_index.append(abs(val-num))
    return collection[np.argmin(best_index)]
