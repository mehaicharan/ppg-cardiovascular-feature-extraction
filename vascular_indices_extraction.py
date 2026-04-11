import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

def extract_vascular_morphology(raw_signal_window, sample_rate):
    signal_array = np.asarray(raw_signal_window).astype(float)    
    # 1. Identify beat boundaries (using valleys)
    valleys, _ = find_peaks(-signal_array, distance=int(sample_rate * 0.4), prominence=0.1 * np.std(signal_array))
    if len(valleys) < 2:
        return signal_array, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    # 2. Baseline correction
    baseline_func = interp1d(valleys, signal_array[valleys], kind='linear', bounds_error=False, fill_value="extrapolate")
    baseline = baseline_func(np.arange(len(signal_array)))
    flat_signal = signal_array - baseline  
    # 3. Normalize amplitude between 0 and 1
    sig_min, sig_max = np.min(flat_signal), np.max(flat_signal)
    if sig_max > sig_min:
        flat_signal = (flat_signal - sig_min) / (sig_max - sig_min)

    systolic_indices = []
    diastolic_indices = []
    reflection_indices = []
    time_gaps_ms = []
    si_surrogate_values = []

    #Shift samples to align APG f-point to PPG diastolic peak
    shift_samples = int(0.05 * sample_rate)  
    # 4. Beat-by-Beat Analysis
    for i in range(len(valleys) - 1):
        start_idx = valleys[i]
        end_idx = valleys[i + 1]
        single_beat = flat_signal[start_idx:end_idx]
        beat_length = len(single_beat)        
        if beat_length < int(sample_rate * 0.4): 
            continue

        # A. Find Systolic Peak (Global maximum of the beat)
        relative_sys_idx = np.argmax(single_beat)
        systolic_height = single_beat[relative_sys_idx]
        if systolic_height <= 0: 
            continue
        # Setup search window for Diastolic Peak
        search_start = relative_sys_idx + int(0.10 * beat_length)
        search_end = min(relative_sys_idx + int(0.60 * beat_length), beat_length - 1)        
        if search_start >= search_end - 5: 
            continue
        relative_dia_idx = None
        search_window = single_beat[search_start:search_end]

        # B. Attempt to find raw diastolic peak
        raw_peaks, raw_props = find_peaks(search_window, prominence=max(0.01 * systolic_height, 1e-4))        
        if len(raw_peaks) > 0:
            best_raw = raw_peaks[np.argmax(raw_props['prominences'])]
            relative_dia_idx = search_start + best_raw            
        else:
            # C. If no raw peak, use 2nd Derivative (APG)
            sg_window = int(0.05 * sample_rate) | 1 # Ensure odd number
            if sg_window < 5: sg_window = 5           
            try:
                apg = savgol_filter(single_beat, window_length=sg_window, polyorder=3, deriv=2)
            except ValueError:
                apg = np.gradient(np.gradient(single_beat))         
            search_apg_start = relative_sys_idx + int(0.05 * sample_rate)
            search_apg = apg[search_apg_start:search_end]            
            e_peaks, _ = find_peaks(search_apg, prominence=np.std(search_apg)*0.1)       
            if len(e_peaks) > 0:
                best_e_idx = search_apg_start + e_peaks[0]                
                f_search = apg[best_e_idx:search_end]
                f_valleys, _ = find_peaks(-f_search)                
                if len(f_valleys) > 0:
                    f_idx = best_e_idx + f_valleys[0]
                    relative_dia_idx = f_idx - shift_samples
                else:
                    relative_dia_idx = best_e_idx - shift_samples
            
            if relative_dia_idx is not None and relative_dia_idx <= relative_sys_idx:
                relative_dia_idx = relative_sys_idx + int(0.05 * sample_rate) 

        # 5. Feature Calculation
        if 0 < relative_dia_idx < beat_length:
            diastolic_height = single_beat[relative_dia_idx]          
            systolic_indices.append(start_idx + relative_sys_idx)
            diastolic_indices.append(start_idx + relative_dia_idx)            
            # Reflection Index (RI) = (Diastolic Height / Systolic Height) * 100
            ri_percent = (diastolic_height / systolic_height) * 100.0
            reflection_indices.append(ri_percent)            
            # Peak-to-Peak Time Gap (Delta T)
            delta_t_seconds = (relative_dia_idx - relative_sys_idx) / sample_rate
            time_gaps_ms.append(delta_t_seconds * 1000.0)           
            # Stiffness Index Surrogate = 1 / Delta T
            si_surrogate_value = 1.0 / delta_t_seconds if delta_t_seconds > 0 else 0
            si_surrogate_values.append(si_surrogate_value)

    return (flat_signal, np.array(systolic_indices), np.array(diastolic_indices), 
            np.array(reflection_indices), np.array(time_gaps_ms), np.array(si_surrogate_values))