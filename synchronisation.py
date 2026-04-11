from scipy.signal import correlate
from scipy.interpolate import interp1d
import numpy as np

print("Applying Two-Step Synchronisation...")

# Initial Thump Alignment
# Look for the thump artefact the first 15 seconds (the buffer period)
sync_search_pts = int(15 * TARGET_SAMPLE_RATE)
idx_thump_f = np.argmax(np.abs(finger_sync_filtered[:sync_search_pts]))
idx_thump_w = np.argmax(np.abs(wrist_sync_filtered[:sync_search_pts]))

# Get the exact timestamps of those thump/spikes
finger_thump_time = finger_time_axis[idx_thump_f]
wrist_thump_time = wrist_uniform_time_axis[idx_thump_w]

# Calculate the initial offset and shift the wrist's timeline
initial_time_shift = finger_thump_time - wrist_thump_time
wrist_shifted_initial = wrist_uniform_time_axis + initial_time_shift

# Refinement via Cross-Correlation
refine_start = int(20 * TARGET_SAMPLE_RATE) 
refine_end = int(30 * TARGET_SAMPLE_RATE)

# Temporarily compare the wrist and finger signal mathematically 
temp_wrist_interp = interp1d(wrist_shifted_initial, wrist_sync_filtered, bounds_error=False, fill_value=0)
wrist_temp_synced = temp_wrist_interp(finger_time_axis)

# Isolate the later segment for both signals
seg_f = finger_sync_filtered[refine_start:refine_end]
seg_w = wrist_temp_synced[refine_start:refine_end]

# Perform discrete cross-correlation
corr = correlate(seg_f, seg_w, mode='full')

# Find the optimal lag index (m_opt) and convert to time (Delta tau)
optimal_lag_idx = np.argmax(corr) - (len(seg_f) - 1)
refinement_shift = optimal_lag_idx / TARGET_SAMPLE_RATE 

# Final Interpolation and Truncation
# Combine the initial thump shift and the cross-correlation shift
total_time_shift = initial_time_shift + refinement_shift
wrist_final_time_axis = wrist_uniform_time_axis + total_time_shift

# Interpolate the RAW wrist signal onto the finger's exact timeline using the final shift
wrist_raw_synced = interp1d(wrist_final_time_axis, wrist_resampled_raw, bounds_error=False, fill_value=0)(finger_time_axis)

# Trim both arrays to their minimum shared length
min_len = min(len(finger_raw_signal), len(wrist_raw_synced))
finger_raw_synced = finger_raw_signal[:min_len]
wrist_raw_synced = wrist_raw_synced[:min_len]

# Establish the single, common time base for all downstream analysis
master_time_axis = finger_time_axis[:min_len]
print("Synchronization complete. Both signals are now phase-locked on a common time base.")