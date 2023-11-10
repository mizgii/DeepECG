from  scipy.signal import butter, filtfilt, iirnotch


def filter_signal(signal, fs=500):

    lowcut = 0.5  # Lower cutoff frequency in Hz

    [b, a] = butter(3, lowcut, btype='highpass')
    signal = filtfilt(b, a, signal, axis=0)
    [bn,an] = iirnotch(50, 30, fs=fs)
    signal = filtfilt(bn, an, signal, axis=0)

    return signal

