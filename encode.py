import numpy as np

from util import chunk_signal
from util import SignalType

def zero_crossings(sig: np.ndarray):
    z = 0
    for i in range(1, len(sig)):
        if sig[i-1] > 0 and sig[i] < 0 or sig[i-1] < 0 and sig[i] > 0:
            z += 1
    return z


def autocorrelate(sig):
    pass

def classify_voicing(sig, sr):
    """ Classify whether a given portion of a signal is voiced or unvoiced.
        Likely works best on small signal chunks
        0 - silent
        1 - unvoiced
        2 - voiced

        We use the distributions provided in Pattern recognition approach to voiced-unvoiced-silence classification with applications to speech recognition
        by Bishnu and Rabiner (IEEE)    
    """

    # For now, just use the mean of the distribution
    threshold_10_msec = 30 
    z = zero_crossings(sig)

    if z > threshold:
        return SignalType.UNVOICED
    else:
        return SignalType.VOICED
    
    # zero crossings?

    # autocorrelation?
    pass

# def find_fundamental_frequency():
#     pass

def analyze_lpc(sig: np.ndarray, p: int):
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

    lpc_coeffs = np.linalg.lstsq(A, b)[0]

    b_ = np.matmul(A, lpc_coeffs.reshape(-1, 1))

    lpc_coeffs *= -1 # We do this to match the form of the transfer function used in scipy lfilter
    residuals = b - np.squeeze(b_)
    
    return lpc_coeffs, residuals
 
def analyze_signal(sig: np.ndarray, sr: int, npoles: int, chunk_size_msec: float, chunk_offset_ratio: float = 1.0):
    """
        sig - numpy array representing the signal for which to extract lpc params
        sr - sampling ratio
        npoles - The number of poles in the transfer function
        chunk_size_msec - width of each chunk over which lpc will be run
        chunk_offset_ratio - offset between successive chunks as a function of chunk size
        annotate_voicing - If set to True, this will classify each signal chunk as voiced, unvoiced, or silent
    """
    slices, chunks = chunk_signal(sig, sr, chunk_size_msec, chunk_offset_ratio)
    
    # Split the signal into chunks and return the lpc params for each chunk
    lpc_params = []
    for chunk in chunks:
        a, e = analyze_lpc(chunk, npoles)
        v = classify_voicing(chunk)
        lpc_params.append((a,e,v))

    return lpc_params, chunk_size_msec, chunk_offset_ratio