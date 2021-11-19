import scipy.fftpack as fft
import numpy as np
from timeit import default_timer as timer
import pyfftw
import pandas as pd
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT


def mflops(N, time_ms):
    return 2.5 * N * np.log2(N) / time_ms

#length = [2**x for x in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
length = [2**x for x in range(18)]
#length.append(100)
#length.append(1000)
#length.append(1023)

def clfft(data_host):
  data_gpu = cla.to_device(queue, data_host)
  transform = FFT(context, queue, data_gpu)
  event, = transform.enqueue()
  event.wait()
  result_host = data_gpu.get()
  return result_host

def clfft_nocopy(data_host):
  transform = FFT(context, queue, data_gpu)
  event, = transform.enqueue()
  event.wait()
  result_host = data_gpu.get()
  return result_host

def time_fft(func, data):
    start = timer()
    for i in range(1000):
        res = func(data)
    end = timer()
    return end - start

result = pd.DataFrame(columns=['scipy', 'numpy', 'pyfftw', 'pyfftw-aligned'])


print(length)
def test_scipy():
    for l in length:
        sigin = np.zeros(l, dtype=np.float)
        result.loc[l, 'scipy'] = time_fft(fft.fft, sigin)
        result.loc[l, 'scipy.rfft'] = time_fft(fft.rfft, sigin)

def test_numpy():
    for l in length:
        sigin = np.zeros(l, dtype=np.float)
        result.loc[l, 'numpy'] = time_fft(np.fft.fft, sigin)
        result.loc[l, 'numpy.rfft'] = time_fft(np.fft.rfft, sigin)

def test_pyfftw():
    for l in length:
        sigin = np.zeros(l, dtype=np.float)
        result.loc[l, 'pyfftw'] = time_fft(pyfftw.interfaces.numpy_fft.fft, sigin)
        sigin = pyfftw.empty_aligned(l, dtype='float')
        sigin[:] = np.zeros(l)
        result.loc[l, 'pyfftw-aligned'] = time_fft(pyfftw.interfaces.numpy_fft.fft, sigin)


result = pd.DataFrame(columns=['scipy', 'numpy', 'pyfftw', 'pyfftw-aligned', 'gpyfft', 'gpyfft-nocopy'])
def test_complex_scipy():
    for l in length:
        sigin = np.zeros(l, dtype = np.complex64)
        result.loc[l, 'scipy'] = time_fft(fft.fft, sigin)

def test_complex_numpy():
    for l in length:
        sigin = np.zeros(l, dtype = np.complex64)
        result.loc[l, 'numpy'] = time_fft(np.fft.fft, sigin)

def test_complex_gpyfft():
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    for l in length:
        sigin = np.zeros(l, dtype = np.complex64)
        result.loc[l, 'gpyfft'] = time_fft(clfft, sigin)
        data_gpu = cla.to_device(queue, sigin)
        result.loc[l, 'gpyfft-nocopy'] = time_fft(clfft_nocopy, sigin)
def test_complex_pyfftw():
    for l in length:
        sigin = np.zeros(l, dtype = np.complex64)
        result.loc[l, 'pyfftw'] = time_fft(pyfftw.interfaces.numpy_fft.fft, sigin)
        sigin = pyfftw.empty_aligned(l, dtype='complex64')
        sigin[:] = np.zeros(l)
        result.loc[l, 'pyfftw-aligned'] = time_fft(pyfftw.interfaces.numpy_fft.fft, sigin)


