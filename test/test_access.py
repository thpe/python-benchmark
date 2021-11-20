import numpy as np
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
import pytest

def npsetzero(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = 0

def npsetzerocol(data):
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            data[j][i] = 0

def pdsetzeroloc(data):
    for col in data.columns:
        for row in data.index:
            data.loc[row, col] = 0

def pdsetzeroiloc(data):
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            data.iloc[row, col] = 0

def pdtonp(df):
    data = np.array(df)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            data[j][i] = 0

def time_func(func, data, count=100):
    start = timer()
    for i in range(int(count)):
        res = func(data)
    end = timer()
    return (end - start) / count

length = [2**x for x in range(8)]
result = pd.DataFrame(columns=['numpyrow'])
def test_numpyrow():
    for l in tqdm(length):
      a = np.zeros((l,200))
      df = pd.DataFrame(a)
      result.loc[l, 'numpyrow'] = time_func(npsetzero, a, 5)
def test_numpycol():
    for l in tqdm(length):
      a = np.zeros((l,200))
      df = pd.DataFrame(a)
      result.loc[l, 'numpycol'] = time_func(npsetzerocol, a, 5)
def test_pdsetzeroloc():
    for l in tqdm(length):
      a = np.zeros((l,200))
      df = pd.DataFrame(a)
      result.loc[l, 'pandasloc'] = time_func(pdsetzeroloc, df, 5)
def test_pdsetzeroiloc():
    for l in tqdm(length):
      a = np.zeros((l,200))
      df = pd.DataFrame(a)
      result.loc[l, 'pandasiloc'] = time_func(pdsetzeroiloc, df, 5)

def test_pdtonp():
    for l in tqdm(length):
      a = np.zeros((l,200))
      df = pd.DataFrame(a)
      result.loc[l, 'pdtonp'] = time_func(pdsetzeroiloc, df, 5)
def test_dumpdf():
    result.to_csv('result.csv')
    with open('result.md', 'w') as f:
      result.to_markdown(f)
