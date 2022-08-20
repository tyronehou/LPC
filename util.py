from enum import Enum

import scipy
import numpy as np

class SignalType(Enum):
    SILENT = 0
    UNVOICED = 1
    VOICED = 2

def load_signal(filename: str, online: bool = False):
    """ Load signal as an int 16 numpy array
    """
    if online:
        raise NotImplementedError("Please use offline mode for the time being")

    sr, sig = scipy.io.wavfile.read(filename)

    if sig.dtype == np.float32:
        max_amp = np.iinfo(np.int16).max
        sig = (sig*max_amp).astype(np.int16)
    return sr, sig

def generate_impulse_train(f_hz, sr, duration_secs):
    T = sr // f_hz
    x = np.zeros(duration_secs * sr, dtype=np.int16)
    
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

def time_to_sample(t_sec, sr):
    return int(t_sec * sr)

def chunk_signal(sig: np.ndarray, sr: int, chunk_size_msec: int, chunk_offset_ratio: float = 1.0):
    # Add zeroes to the last chunk if needed?
    #0 + [0, chunk_size_samp]
    #chunk_offset_samp
    #2
    #3

    chunk_size_samp = int(chunk_size_msec / 1000 * sr)
    chunk_offset_samp = int(chunk_size_samp * chunk_offset_ratio)

    # the padding that needs to be added due to the signal not matching the chunk
    z = (len(sig)-chunk_size_samp) % chunk_offset_samp

    if z != 0:
        sig = np.pad(sig, (0, z))
  
    # Through a broadcasting trick we can create a large array of indices
    indices = np.arange(0, len(sig)-chunk_size_samp+1, chunk_offset_samp).reshape(-1, 1) + np.arange(0, chunk_size_samp)

    slices = indices[:, [0, -1]]
    return slices, sig[indices]