### LPC.py
### Author: Tyrone Hou
### Desc: Implementation of LPC algorithm closely following https://ccrma.stanford.edu/~hskim08/lpc/ and taking tips from https://www.kuniga.me/blog/2021/05/13/lpc-in-python.html

import matplotlib.pyplot as plt
import numpy as np
import scipy
import librosa

def generate_impulse_train(freq_hz, sr, duration_secs):
    T = sr // freq_hz
    x = np.zeros(duration_secs * sr, dtype=np.int16)
    #import pdb; pdb.set_trace()

    for i in range(0, len(x), T):
        x[i] = 1

    return x

def generate_sin(freq_hz, sr, duration_secs):
     t = np.arange(int(sr*duration_secs)) / sr
     return np.sin(2.*np.pi*freq_hz*t) * np.iinfo(np.int16).max

def signal_power(sig):
    """ Calculates the signal power as the sum of the absolute squares of the signal divided by signal length"""
    sig = np.array(sig)
    return np.sum(np.abs(sig.astype(np.float32)**2)) / len(sig)

def load_signal(filename, online=False):
    if online:
        raise NotImplementedError("Please use offline mode for the time being")

    sr, sig = scipy.io.wavfile.read(filename)

    if sig.dtype == np.float32:
        max_amp = np.iinfo(np.int16).max
        sig = (sig*max_amp).astype(np.int16)
    return sr, sig

def chunk_signal(sig, chunk_size_samp, chunk_offset_samp=0):
    # Add zeroes to the last chunk if needed?
    #0 + [0, chunk_size_samp]
    #chunk_offset_samp
    #2
    #3

    if chunk_offset_samp == 0:
        chunk_offset_samp = chunk_size_samp 

    # the padding that needs to be added due to the signal not matching the chunk
    z = (len(sig)-chunk_size_samp) % chunk_offset_samp

    if z != 0:
        sig = np.pad(sig, (0, z))
  
    #slices = np.arange().reshape(-1, 1) + np.array([0, chunk_size_samp])
    # Through a broadcasting trick we can create a large array of indices
    indices = np.arange(0, len(sig)-chunk_size_samp+1, chunk_offset_samp).reshape(-1, 1) + np.arange(0, chunk_size_samp)


    slices = indices[:, [0, -1]]
    return slices, sig[indices]

def analyze_lpc(sig, p):
    # b = A* a + e[n]
    # where b is the vector of output predictions, A is a matrix of previous inputs, a is a vector of pole coefficients and e is the residual sequence
    # For a given index, we have b_i = A_i * a + e_i, which maps to single prediction
    
    # The params represent the coefficients of the transfer function denominator, where the coefficients are summed, not subtracted
    N = len(sig)-p

    assert N >= p # assert you have at least p equations for the least squares to sample
    
    # For now assume the stride is 1
    b = sig[p:]
    A = np.empty((N, p), dtype=sig.dtype)
    for i in range(N):
        A[i] = sig[i:i+p][::-1]

    lpc_params, e_sse, _, _ = np.linalg.lstsq(A, b)

    b_ = np.matmul(A, lpc_params.reshape(-1, 1))

    lpc_params *= -1
    #import pdb; pdb.set_trace()
    residuals = b - np.squeeze(b_)
    
    return lpc_params, residuals, b_, e_sse

def synthesize_lpc(lpc_params, source_sig):
    # Apply the all pole filter to the source signal
    lpc_params *= -1
    output_sig = scipy.signal.lfilter([1], [-1, *lpc_params], source_sig)

    return output_sig

# For multiple lpc chunks
# Assume each chunk is the same length
def combine_sig(chunks, chunk_offset, window=None):
    chunk_size = len(chunks[0])
    N = chunk_size + (len(chunks)-1) * chunk_offset
    sig = np.zeros(N)

    if window == None:
        window = np.ones(chunk_size)

    for i, chunk in enumerate(chunks):
        start = i*chunk_offset
        sig[start:start+chunk_size] += chunk * window

    return sig

def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])
    
    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i 
        X[i, :] = xz[offset : offset + p]
    return X

"""
An implementation of LPC.

A detailed explanation can be found at
https://ccrma.stanford.edu/~hskim08/lpc/

x - a vector representing the time-series signal
p - the polynomial order of the all-pole filter

a - the coefficients to the all-pole filter
g - the variance(power) of the source (scalar)
e - the full error signal

NOTE: This is not the most efficient implementation of LPC.
Matlab's own implementation uses FFT to via the auto-correlation method
which is noticeably faster. (O(n log(n)) vs O(n^2))
"""
def solve_lpc(x, p):
    b = x[1:].T
        
    X = make_matrix_X(x, p)
    
    a, e_sse, _, _ = np.linalg.lstsq(X, b)

    #import pdb; pdb.set_trace()
    e = b.T - np.dot(X, a)
    g = np.var(e)

    return [a, g]

def run_lpc(filename, npoles, chunk_size_sec, chunk_offset_sec=0, online=False):
    sr, sig = load_signal(filename, online)

    chunk_size_samp = int(chunk_size_sec * sr)
    chunk_offset_samp = int(chunk_offset_sec * sr)

    slices, chunks = chunk_signal(sig, chunk_size_samp, chunk_offset_samp)

    #return sr, sig, slices, chunks
    max_amp = np.iinfo(np.int16).max

    source_sigs = []
    outputs = []

    for chunk in chunks:
        a, e, _, _ = analyze_lpc(chunk, npoles)
        #a, g = solve_lpc(chunk, npoles)
        source_sig = np.random.rand(chunk_size_samp) * 2 * max_amp - max_amp

        source_power = signal_power(source_sig)
        source_sig *= np.sqrt(np.var(e) / source_power)
        #source_sig *= np.sqrt(g / source_power)
        #import pdb; pdb.set_trace()
        outputs.append(synthesize_lpc(a, source_sig.astype(np.int16)))
        source_sigs.append(source_sig)
    #import pdb; pdb.set_trace()
    return sr, combine_sig(outputs, chunk_offset_samp), combine_sig(source_sigs, chunk_offset_samp)


if __name__ == '__main__':
    sigfile = "signals/in/BabyElephantWalk60.wav" 
    sr, sig, slices, chunks = run_lpc(sigfile, 6, 1, 0.3, online=False)

    print(sig.shape)