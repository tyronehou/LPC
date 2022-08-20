import scipy
import numpy as np

from util import generate_impulse_train
from util import generate_sin

def run_source_filter(lpc_coeffs, source_sig=None):
    # Apply the all pole filter to the source signal
    output_sig = scipy.signal.lfilter([1], [1, *lpc_coeffs], source_sig)

    return output_sig

def synthesize_lpc(signal_type, lpc_params):
	"""
		Generate an appropriate source signal and run it through a filter
	"""
    if ff: # If fundamental frequency is passed, the signal is assumed to be voiced
        source_sig = ;
    else: #
        source_sig = 

    run_source_filter(lpc_coeffs, source_sig)


# For multiple lpc chunks
# Assume each chunk is the same length
def combine_signal(chunks, chunk_offset_samp, window=None):
	"""
		Combine signal chunks of the same length
		chunks - list of signal chunks
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
		pass
		if v == 0:
		elif v == 1:
		elif v == 2:
		else:

		# Determine voice/unvoiced
			# Generate source sig
		# synthesize_lpc
		# append to chunks
	
	output_sig = combine_signal(signal_chunks, chunk_offset_samp)

	return output_sig