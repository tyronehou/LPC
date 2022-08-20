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
 
        Unvoiced mean: 49.914
        Voiced mean: 12.775
    """

    # For now, just use the approximate midpoint of the means of the unvoiced/voiced distribution
    # Not the best determination, but will do for a first pass
    threshold_10_msec = 30
    sig_len_sec = len(sig) / sr
    z = zero_crossings(sig) * 0.010 / sig_len_msec

    if z > threshold:
        return SignalType.UNVOICED
    else:
        return SignalType.VOICED
    
    # autocorrelation?