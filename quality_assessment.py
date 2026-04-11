import numpy as np
from preprocessing import TARGET_SAMPLE_RATE
from scipy.signal import find_peaks

def evaluate_window_quality(signal_window, sample_rate, window_duration_sec=10.0, quality_threshold=0.7):
    window_std = np.std(signal_window)
    
    # Check 1: Reject flat-line segments
    if window_std < 1e-6:
        return False
    # Check 2: Peak Detection (0.4s distance, 0.2*sigma prominence)
    min_distance_samples = int(sample_rate * 0.4)
    prominence_thresh = 0.20 * window_std
    pos_peaks, _ = find_peaks(signal_window, distance=min_distance_samples, prominence=prominence_thresh)
    neg_peaks, _ = find_peaks(-signal_window, distance=min_distance_samples, prominence=prominence_thresh)
    # Check 3: Minimum peak-count criterion (80% of expected at 50 bpm)
    min_expected_hr = 50.0
    expected_beats = (min_expected_hr / 60.0) * window_duration_sec
    min_required_peaks = int(0.8 * expected_beats) # 6 peaks for a 10s window

    if len(pos_peaks) < min_required_peaks:
        return False
    pos_amplitudes = signal_window[pos_peaks]
    neg_amplitudes = -signal_window[neg_peaks]

    # Require at least 3 peaks to compute meaningful dispersion statistics
    if len(pos_amplitudes) < 3 or len(neg_amplitudes) < 3:
        return False
    mean_pos = np.mean(pos_amplitudes) + 1e-8
    mean_neg = np.mean(neg_amplitudes) + 1e-8
    
    # Coefficient of variation for positive (D1) and negative (D2) peaks
    D1 = np.std(pos_amplitudes) / mean_pos
    D2 = np.std(neg_amplitudes) / mean_neg

    # Composite quality score (Q) and accept if Q > 0.7
    Q_score = np.exp(-(abs(D1) + abs(D2)))
    return Q_score > quality_threshold

def build_joint_quality_mask(finger_signal, wrist_signal, sample_rate, window_duration_sec=10.0, quality_threshold=0.7):
    print("Assessing signal quality and building joint validity mask...")
    
    window_samples = int(window_duration_sec * sample_rate)
    step_samples = int(window_samples * 0.5) # 50% overlap
    
    # Initialize boolean masks (defaulting to False/Bad)
    finger_mask = np.zeros(len(finger_signal), dtype=bool)
    wrist_mask = np.zeros(len(wrist_signal), dtype=bool)
    
    # Slide the 10s window across the signals
    for start_idx in range(0, len(finger_signal) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples      
        finger_window = finger_signal[start_idx:end_idx]
        wrist_window = wrist_signal[start_idx:end_idx]        
        finger_is_valid = evaluate_window_quality(finger_window, sample_rate, window_duration_sec, quality_threshold)
        wrist_is_valid = evaluate_window_quality(wrist_window, sample_rate, window_duration_sec, quality_threshold)
        
        # If the window passes, mark all samples within that window as True(Valid)
        if finger_is_valid:
            finger_mask[start_idx:end_idx] = True
        if wrist_is_valid:
            wrist_mask[start_idx:end_idx] = True
            
    # "A joint mask was then formed using the logical AND of both"
    joint_valid_mask = finger_mask & wrist_mask
    
    valid_percentage = (np.sum(joint_valid_mask) / len(joint_valid_mask)) * 100
    print(f"Quality assessment complete: {valid_percentage:.1f}% of the synchronized timeline is jointly valid.")
    return joint_valid_mask

# Create filtered versions of the final, truncated, and aligned signals specifically for quality checking
finger_quality_signal = apply_hybrid_filter(finger_raw_synced, TARGET_SAMPLE_RATE)
wrist_quality_signal = apply_hybrid_filter(wrist_raw_synced, TARGET_SAMPLE_RATE)

# Generate the joint mask
joint_quality_mask = build_joint_quality_mask(finger_quality_signal, wrist_quality_signal, TARGET_SAMPLE_RATE)