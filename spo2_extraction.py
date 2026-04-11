import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks

def extract_dc_component(signal_array, sample_rate):
    if np.std(signal_array) == 0: 
        return np.zeros_like(signal_array)   
    nyquist_freq = 0.5 * sample_rate
    b, a = butter(4, 0.5 / nyquist_freq, btype='low') 
    return filtfilt(b, a, signal_array)

def extract_ac_component(signal_array):
    if np.std(signal_array) == 0: 
        return np.zeros_like(signal_array)    
    wavelet_level = 8 
    wavelet_coeffs = pywt.wavedec(signal_array, 'db4', level=wavelet_level)    
    # Remove high-frequency noise (detail coefficients)
    for i in range(1, 6):
        wavelet_coeffs[-i] = np.zeros_like(wavelet_coeffs[-i])        
    # Remove DC baseline (approximation coefficients)
    wavelet_coeffs[0] = np.zeros_like(wavelet_coeffs[0])   
    return pywt.waverec(wavelet_coeffs, 'db4')[:len(signal_array)]

def apply_hampel_filter(spo2_array, window_size=5, n_sigmas=2.0):
    if len(spo2_array) == 0: 
        return spo2_array   
    spo2_clean = np.copy(spo2_array)
    num_points = len(spo2_clean)    
    for i in range(num_points):
        start_idx = max(0, i - window_size)
        end_idx = min(num_points, i + window_size)        
        local_window = spo2_clean[start_idx:end_idx]
        local_median = np.median(local_window)
        mad = np.median(np.abs(local_window - local_median))      
        # If the point is a severe outlier, snap it to the local median
        if mad != 0 and np.abs(spo2_clean[i] - local_median) > (n_sigmas * mad):
            spo2_clean[i] = local_median            
    return spo2_clean

def extract_spo2_beat_to_beat(raw_red_signal, raw_ir_signal, sample_rate, spo2_a=110.0, spo2_b=25.0):
    # 1. Extract DC Components from the raw signals
    dc_red_signal = extract_dc_component(raw_red_signal, sample_rate)
    dc_ir_signal  = extract_dc_component(raw_ir_signal, sample_rate)  # 2. Extract AC Components (invert raw signals first so systolic peaks point UP)
    ac_red_signal = extract_ac_component(-raw_red_signal)
    ac_ir_signal  = extract_ac_component(-raw_ir_signal)    
    # 3. Peak and Trough Detection on the Infrared Channel
    adaptive_prominence = max(0.20 * np.std(ac_ir_signal), 0.005) 
    min_peak_distance = int(sample_rate * 0.4)   
    peaks, _ = find_peaks(ac_ir_signal, distance=min_peak_distance, prominence=adaptive_prominence)
    troughs, _ = find_peaks(-ac_ir_signal, distance=min_peak_distance, prominence=adaptive_prominence)
 
    raw_spo2_values = []
    valid_beat_indices = []   
    # 4. Beat-to-Beat Ratio of Ratios Calculation
    for peak_idx in peaks:
        # Find the immediately preceding trough
        preceding_troughs = troughs[troughs < peak_idx]
        if len(preceding_troughs) == 0: 
            continue
        trough_idx = preceding_troughs[-1]        
        # AC is calculated as the trough-to-peak difference
        ac_red_amplitude = ac_red_signal[peak_idx] - ac_red_signal[trough_idx]
        ac_ir_amplitude = ac_ir_signal[peak_idx] - ac_ir_signal[trough_idx]       
        # DC is strictly sampled at the trough location
        dc_red_baseline = dc_red_signal[trough_idx]
        dc_ir_baseline = dc_ir_signal[trough_idx]        
        # Reject non-positive AC/DC values
        if dc_red_baseline <= 0 or dc_ir_baseline <= 0 or ac_ir_amplitude <= 0 or ac_red_amplitude <= 0: 
            continue        
        # Reject weak perfusion (Perfusion Index < 0.1%)
        perfusion_index_ir = (ac_ir_amplitude / dc_ir_baseline) * 100.0
        if perfusion_index_ir < 0.1: 
            continue            
        # Compute Ratio of Ratios (R)
        ratio_of_ratios = (ac_red_amplitude / dc_red_baseline) / (ac_ir_amplitude / dc_ir_baseline)       
        # Convert to SpO2 using empirical linear calibration
        spo2_estimate = spo2_a - (spo2_b * ratio_of_ratios)        
        # Retain only physiological values
        if 70.0 <= spo2_estimate <= 100.0:
            raw_spo2_values.append(spo2_estimate)
            valid_beat_indices.append(peak_idx)          
    # 5. Apply Hampel filter to suppress outliers
    clean_spo2_values = apply_hampel_filter(np.array(raw_spo2_values), window_size=5)   
    return clean_spo2_values, np.array(valid_beat_indices)