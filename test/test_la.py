import numpy as np
import pytest
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm

def matmul(data):
    return np.matmul(data[0], data[1])

def time_func(func, data):
    start = timer()
    for i in range(100):
        res = func(data)
    end = timer()
    return end - start


length = [2**x for x in range(12)]
result = pd.DataFrame(columns=['numpy'])

def test_matmul():
    for l in tqdm(length):
      a = np.zeros((l,l))
      b = np.zeros((l,l))
      result.loc[l, 'numpy'] = time_func(matmul, (a, b))
def test_dump():
    result.to_csv('result_la.csv')
    with open('result_la.md', 'w') as f:
        result.to_markdown(f)
