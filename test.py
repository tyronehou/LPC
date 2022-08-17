import unittest

import scipy
import matplotlib.pyplot as plt

from LPC import *
import librosa

def test_signal_power():
    s = [1, 2, 3, 4]
    unittest.assertEqual(signal_power(s), 7.5)

def test_run_lpc():
    filename = "creak.wav"
    #filename = "bubbles.wav"
    #filename = "ah1.wav"
    p = 6

    sr, y, source = run_lpc(f"signals/in/{filename}", p, 0.005, 0.005, online=False)
    #import pdb; pdb.set_trace()
    scipy.io.wavfile.write(f"signals/out/lpc_{filename}", sr, y.astype(np.int16))
    scipy.io.wavfile.write(f"signals/out/source_{filename}", sr, source.astype(np.int16))

def test_combine_sig():
    chunks = [[1,2,3],[4,5,6],[7,8,9]]
    chunk_offset = 3
    combined_sig = combine_sig(chunks, chunk_offset)
    print(combined_sig)
    #self.assertEqual(combined_sig, [1,2,7,5,13,8,9])

def test_lpc():
    max_amp = np.iinfo(np.int16).max

    filename = "bubbles.wav"
    #filename = "ah1.wav"
    p = 6
    f_hz = 400
    t_secs = 1

    sr, sig = load_signal(f"signals/in/{filename}")
    
    print(f"sampling rate: {sr}")
    
    sig = sig[sr:2*sr]
    a, e, b_, e_sse = analyze_lpc(sig, p)

    #a_ = librosa.lpc(sig / max_amp, p)
    #a, g = solve_lpc(sig, p)

    #e_sig = generate_impulse_train(f_hz, sr, t_secs)

    e_sig = np.random.randint(-max_amp, max_amp, size=sr*t_secs)
    source_sig = np.random.rand(sr) * 2 * max_amp - max_amp

    orig_power = signal_power(sig)
    e_power = signal_power(e_sig)

    e_sig = e_sig * np.var(e) / orig_power
    #e_sig = (e_sig * np.sqrt(orig_power / e_power))

    y = synthesize_lpc(a, e_sig)

    print(f"Original signal power: {orig_power}, Source synth signal power: {e_power}")
    y = (y * np.sqrt(orig_power / e_power))

    y = y.astype(np.int16)

    scipy.io.wavfile.write(f"signals/out/residual_{filename}", sr, e.astype(np.int16))
    scipy.io.wavfile.write(f"signals/out/source_{filename}", sr, e_sig.astype(np.int16))
    scipy.io.wavfile.write(f"signals/out/lpc_{filename}", sr, y.astype(np.int16))


    print(f"LPC params: {a}; residual sum of squared error: {e_sse}")

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(sig)
    axes[1].plot(e)
    axes[2].plot(y)
    plt.show()


if __name__ == '__main__':
    test_run_lpc()
    #test_lpc()