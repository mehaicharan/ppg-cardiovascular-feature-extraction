import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend

def extract_breathing_rate(raw_signal_window, sample_rate):
    if np.std(raw_signal_window) == 0: 
        return 0.0    
    nyquist_freq = 0.5 * sample_rate
    # 1. Respiratory Bandpass Filter (0.083 Hz to 0.50 Hz)
    b, a = butter(2, [0.083 / nyquist_freq, 0.50 / nyquist_freq], btype='band')
    respiratory_band_signal = filtfilt(b, a, raw_signal_window)   
    # 2. Detrending
    detrended_signal = detrend(respiratory_band_signal)    
    # 3. Spectral Analysis via Welch's Method
    num_points = len(detrended_signal)
    freqs, psd = welch(detrended_signal, sample_rate, nperseg=num_points, nfft=num_points * 10)    
    # 4. Restrict search to physiological target range (0.083 Hz to 0.5 Hz)
    valid_indices = np.where((freqs >= 0.083) & (freqs <= 0.5))[0]    
    if len(valid_indices) == 0: 
        return 0.0    
    # 5. Extract dominant frequency and convert Hz to br/min
    dominant_freq_hz = freqs[valid_indices][np.argmax(psd[valid_indices])]    
    breathing_rate_bpm = dominant_freq_hz * 60.0   
    return breathing_rate_bpm