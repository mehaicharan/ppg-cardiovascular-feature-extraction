import numpy as np
from scipy.signal import find_peaks

def extract_hr_and_hrv(filtered_signal_window, sample_rate, quality_mask_window):
    # Pad the mask with False at both ends to easily find transitions
    padded_mask = np.concatenate(([False], quality_mask_window, [False]))
    segment_starts = np.where(np.diff(padded_mask.astype(int)) == 1)[0]
    segment_ends = np.where(np.diff(padded_mask.astype(int)) == -1)[0]
    valid_peak_indices = []
    clean_rr_intervals_ms = []
    
    # Iterate through each continuous valid segment defined by the quality mask
    for start_idx, end_idx in zip(segment_starts, segment_ends):      
        # Exclude segments shorter than 3 seconds (cannot reliably calculate intervals)
        if (end_idx - start_idx) < (3 * sample_rate):
            continue
        local_segment = filtered_signal_window[start_idx:end_idx]     
        adaptive_prominence = max(0.25 * np.std(local_segment), 0.001)
        min_peak_distance = int(sample_rate * 0.4) 
        local_peaks, _ = find_peaks(local_segment, distance=min_peak_distance, prominence=adaptive_prominence)   
        global_peaks = local_peaks + start_idx
        valid_peak_indices.extend(global_peaks)   
        
        # Calculate RR intervals ONLY within this continuous valid segment
        if len(local_peaks) > 1:
            segment_rr_intervals = np.diff(global_peaks) / sample_rate * 1000.0           
            # Enforce physiological limits: 250ms (240 bpm) to 2000ms (30 bpm)
            valid_rr_segment = segment_rr_intervals[(segment_rr_intervals >= 250) & (segment_rr_intervals <= 2000)]
            clean_rr_intervals_ms.extend(valid_rr_segment)
    valid_peak_indices = np.array(valid_peak_indices)
    clean_rr_intervals_ms = np.array(clean_rr_intervals_ms)
   
    # If not enough valid beats were found to calculate HR/HRV, return zeros
    if len(clean_rr_intervals_ms) < 2: 
        return 0.0, {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}, valid_peak_indices, np.array([])
    rr_successive_differences = np.diff(clean_rr_intervals_ms)
    # Calculate HR and HRV
    median_hr_bpm = 60000.0 / np.median(clean_rr_intervals_ms)   
    hrv_metrics = {
        "SDNN": np.std(clean_rr_intervals_ms),
        "RMSSD": np.sqrt(np.mean(np.square(rr_successive_differences))) if len(rr_successive_differences) > 0 else 0.0,
        "pNN50": (100.0 * np.sum(np.abs(rr_successive_differences) > 50) / len(rr_successive_differences)) if len(rr_successive_differences) > 0 else 0.0
    }    
    return median_hr_bpm, hrv_metrics, valid_peak_indices, clean_rr_intervals_ms