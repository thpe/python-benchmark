import numpy as np
import pytest
import scipy
from scipy import linalg
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm

def matmul(data):
    return np.matmul(data[0], data[1])

def np_matinv(data):
    return np.linalg.inv(data)

def sp_matinv(data):
    return linalg.inv(data)

def time_func(func, data):
    start = timer()
    for i in range(100):
        res = func(data)
    end = timer()
    return end - start


length = [2**x for x in range(12)]
length += [11 * x for x in range(1,12)]
length.sort()
result = pd.DataFrame(columns=['np'])

def test_matmul():
    for l in tqdm(length):
      a = np.zeros((l,l))
      b = np.zeros((l,l))
      result.loc[l, 'np'] = time_func(matmul, (a, b))

def test_dump():
    result.to_csv('result_la_mul.csv')
    with open('result_la_mul.md', 'w') as f:
        result.to_markdown(f)

result = pd.DataFrame(columns=['np'])

def test_inv():
    for l in tqdm(length):
      a = np.identity(l)
      result.loc[l, 'np_id'] = time_func(np_matinv, a)
      result.loc[l, 'sp_id'] = time_func(sp_matinv, a)
      a = np.random.random((l,l))
      result.loc[l, 'np'] = time_func(np_matinv, a)
      result.loc[l, 'sp'] = time_func(sp_matinv, a)

def test_dump1():
    result.to_csv('result_la_inv.csv')
    with open('result_la_inv.md', 'w') as f:
        result.to_markdown(f)
