import scipy
import numpy as np

from util import generate_impulse_train
from util import generate_sin
from util import SignalType

def run_source_filter(lpc_coeffs, source_sig=None):
    # Apply the all pole filter to the source signal
    output_sig = scipy.signal.lfilter([1], [1, *lpc_coeffs], source_sig)

    return output_sig

def synthesize_lpc(sr, duration_sec, params):
    """
        Generate an appropriate source signal and run it through a filter
        sr - sampling rate
        duration_sec - output signal duration in secs
        params - a tuple of parameters needed to reconstruct the signal. Will be one of 3 forms depending on the signal type (see below)
            The parameters must be one of the following values:

            v - signal type
            a - vector of lpc_coeffs, the first index corresponds to the coefficient for sig[n-1] (e.g first to be passed to lfilter)
            g - residual signal power/variance
            ff - fundamental frequency

            (SignalType.SILENT,)
            (SignalType.VOICED, a, g)
            (SignalType.UNVOICED, a, g, ff)
    """

    max_amp = np.iinfo(np.int16).max()
    duration_samp = int(duration_sec * sr)

    v = params[0]

    if v == SignalType.SILENT:
        return 
    elif v == SignalType.UNVOICED:
        _, a, g = params

        source_sig = np.random.randint(, size=chunk)
        source_sig = np.random.rand(duration_samp) * 2 * max_amp - max_amp
        source_sig *= np.sqrt(sig_power(e) / source_power)
    elif v == SignalType.VOICED:
        _, a, g, ff = params
        
        source_sig = ;
        source_sig = 

    return run_source_filter(lpc_params["lpc_coeffs"], source_sig)

def combine_signal(chunks, chunk_offset_samp, window=None):
    """
        Combine signal chunks of the same length
        chunks - list of equally sized signal chunks
    """
    chunk_size = len(chunks[0])
    N = chunk_size + (len(chunks)-1) * chunk_offset_samp
    sig = np.zeros(N)

    if window == None:
        window = np.ones(chunk_size)

    for i, chunk in enumerate(chunks):
        start = i*chunk_offset_samp
        sig[start:start+chunk_size] += chunk * window

    return sig

def synthesize_signal(lpc_params, chunk_size_msec, chunk_offset_ratio):
    # To synthesisze the signal, we need to know the filter coefficients, the ff for a voiced signal, and the signal power for an unvoiced signal
    chunk_size_samp = int(chunk_size_msec / 1000 * sr)
    chunk_offset_samp = int(chunk_size_samp * chunk_offset_ratio)
    
    chunks = []

    # a is lpc coefficients, e is residual signals
    for a, e, v in lpc_params:
        if v == 0:
            # Calculate
        elif v == 1:
        elif v == 2:
        else:

        # Determine voice/unvoiced
            # Generate source sig
        # synthesize_lpc
        # append to chunks
    
    output_sig = combine_signal(signal_chunks, chunk_offset_samp)

    return output_sig