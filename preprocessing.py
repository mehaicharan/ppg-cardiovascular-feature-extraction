import pandas as pd
import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

FINGER_FILE_PATH = """Input finger file path here"""
WRIST_FILE_PATH = """Input wrist file path here"""
TARGET_SAMPLE_RATE = 500  

# Hybrid Filter Function
def apply_hybrid_filter(signal_array, sample_rate):
    """Wavelet DC removal + 0.5-4.0Hz Bandpass"""
    if np.std(signal_array) == 0: 
        return np.zeros_like(signal_array)   
    # Stage 1: Wavelet DC Removal (db4)
    wavelet_level = int(np.log2(sample_rate))
    wavelet_coeffs = pywt.wavedec(signal_array, 'db4', level=wavelet_level)
    # Remove approximation coefficients (DC baseline)
    wavelet_coeffs[0] = np.zeros_like(wavelet_coeffs[0]) 
    signal_without_dc = pywt.waverec(wavelet_coeffs, 'db4')[:len(signal_array)]
    # Stage 2: Zero-Phase Butterworth Bandpass Filter
    nyquist_freq = 0.5 * sample_rate
    b, a = butter(4, [0.5/nyquist_freq, 4.0/nyquist_freq], btype='band') 
    return filtfilt(b, a, signal_without_dc)

# Data Loading, Inverting, time axis construction and filtering
try:
    # Finger Pipeline
    finger_dataframe = pd.read_csv(FINGER_FILE_PATH, skiprows=5)   
    # Find the correct red channel column
    finger_red_channel_name = next(
        (col for col in finger_dataframe.columns if "Ch : LED 2 (RED)" in 
         col and "Ch : LED 2 (RED) AMBIENT" not in col), 
        finger_dataframe.columns[1]
    )    
    # Load and Invert (-1 multiplier) to make peaks point upwards
    finger_raw_signal = -pd.to_numeric(finger_dataframe[finger_red_channel_name]
                                       , errors='coerce').fillna(0).values    
    # Apply filter
    finger_filtered_signal = apply_hybrid_filter(finger_raw_signal, 
                                                 TARGET_SAMPLE_RATE)   
    # Construct Time Axis (Fixed 500 Hz rate)
    finger_time_axis = np.arange(len(finger_filtered_signal)) / TARGET_SAMPLE_RATE
    # Wrist Pipeline
    wrist_dataframe = pd.read_csv(WRIST_FILE_PATH, skiprows=6)    
    # Find the correct LED column
    wrist_led_channel_name = next(
        (col for col in wrist_dataframe.columns if 'LEDC1' in col and 'tag' 
         not in col.lower()), 
        wrist_dataframe.columns[1]
    )
    wrist_dataframe = wrist_dataframe.dropna(subset=[wrist_led_channel_name, 
                                                     'timestamp'])  
    # Extract epoch timestamps and convert to elapsed time (seconds)
    wrist_raw_timestamps = pd.to_numeric(wrist_dataframe['timestamp'], 
                                         errors='coerce').values
    wrist_raw_timestamps = (wrist_raw_timestamps - wrist_raw_timestamps[0]) / 1000.0    
    # Load and Invert (-1 multiplier)
    wrist_raw_signal = -pd.to_numeric(wrist_dataframe[wrist_led_channel_name], 
                                      errors='coerce').values
    # Construct Time Axis & Resample (Interpolate to uniform 500 Hz grid)
    wrist_uniform_time_axis = np.arange(0, wrist_raw_timestamps[-1], 
                                        1/TARGET_SAMPLE_RATE)
    wrist_interpolator = interp1d(wrist_raw_timestamps, wrist_raw_signal, 
                                  kind='linear', fill_value="extrapolate")
    wrist_resampled_signal = wrist_interpolator(wrist_uniform_time_axis)    
    # Apply filter
    wrist_filtered_signal = apply_hybrid_filter(wrist_resampled_signal, 
                                                TARGET_SAMPLE_RATE)

    print("Successfully loaded, inverted, timelines, and filtered both signals.")
except Exception as e: 
    print(f"File Error: {e}")