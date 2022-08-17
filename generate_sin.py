import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
if __name__ == '__main__':
	 sr= 48000
	 f_hz = int(sys.argv[1])
	 dur_secs = 3 # duration in seconds
	 out_fname = f"sin_{f_hz}hz_{dur_secs}sec.wav"

	 t = np.arange(int(sr*dur_secs)) / sr
	 x = np.sin(2.*np.pi*f_hz*t) * np.iinfo(np.int16).max

	 scipy.io.wavfile.write(f"signals/in/{out_fname}", sr, x.astype(np.int16))
	 scipy.io.wavfile.read(f"signals/in/{out_fname}")