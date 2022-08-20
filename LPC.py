### LPC.py
### Author: Tyrone Hou
### Desc: Implementation of LPC algorithm closely following https://ccrma.stanford.edu/~hskim08/lpc/ and taking tips from https://www.kuniga.me/blog/2021/05/13/lpc-in-python.html
import matplotlib.pyplot as plt
import numpy as np
import scipy

from encode import analyze_signal
from decode import synthesize_signal
from util import load_signal

# from encode import *
# from decode import *
# from util import *

def run_experiment(filename, npoles, chunk_size_msec, chunk_offset_ratio=0, online=False):
    sr, sig = load_signal(filename, online)

    chunk_size_samp = int(chunk_size_msec / 1000 * sr)
    chunk_offset_samp = int(chunk_offset_msec / 1000 * sr)

    # Split the signal into overlapping chunks
    #slices, chunks = chunk_signal(sig, chunk_size_samp, chunk_offset_samp)



    lpc_params = analyze_signal(signal, chunk_size_msec, chunk_offset_ratio)
    output_sig = synthesize_signal(lpc_params, chunk_size_msec, chunk_offset_ratio)

    return sr, output_sig

    # max_amp = np.iinfo(np.int16).max
    # outputs = []
    # # For each chunk, determine te
    # for chunk in chunks:
    #     a, e = analyze_lpc(chunk, npoles)

    #     source_sig = np.random.rand(chunk_size_samp) * 2 * max_amp - max_amp
    #     source_power = signal_power(source_sig)
    #     source_sig *= np.sqrt(np.var(e) / source_power)

    #     outputs.append(synthesize_lpc(a, source_sig.astype(np.int16)))

    # return sr, combine_signal(outputs, chunk_offset_samp)#, combine_signal(source_sigs, chunk_offset_samp)