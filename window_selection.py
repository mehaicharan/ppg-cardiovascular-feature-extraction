# PROTOCOL TIMINGS
from preprocessing import TARGET_SAMPLE_RATE


BUFFER_SEC = 15.0             
RESTING_PHASE_SEC = 300.0     
PACED_BREATHING_SEC = 120.0   

# 1. Define bounds anchored to the synchronisation thump (finger_thump_time)
resting_start_time = finger_thump_time + BUFFER_SEC
resting_end_time = resting_start_time + RESTING_PHASE_SEC
paced_start_time = resting_end_time
paced_end_time = paced_start_time + PACED_BREATHING_SEC

# 2. Select optimal windows independently for each condition
analysis_window_duration = """Input required duration"""

def find_optimal_protocol_window(joint_quality_mask, start_sec, end_sec, target_window_sec, sample_rate):
    start_idx = int(start_sec * sample_rate)
    end_idx = min(int(end_sec * sample_rate), len(joint_quality_mask))
    window_samples = int(target_window_sec * sample_rate)   
    if (end_idx - start_idx) <= window_samples:
        return start_idx, end_idx      
    best_start_idx = start_idx
    max_valid_samples = -1    
    # Slide through the epoch second-by-second to find the highest concentration of 'True' mask values
    for i in range(start_idx, end_idx - window_samples + 1, sample_rate):
        valid_count = np.sum(joint_quality_mask[i : i + window_samples])        
        if valid_count > max_valid_samples:
            max_valid_samples = valid_count
            best_start_idx = i            
    return best_start_idx, best_start_idx + window_samples
    
optimal_resting_start_idx, optimal_resting_end_idx = find_optimal_protocol_window(joint_quality_mask, resting_start_time, resting_end_time, analysis_window_duration, TARGET_SAMPLE_RATE)

optimal_paced_start_idx, optimal_paced_end_idx = find_optimal_protocol_window(joint_quality_mask, paced_start_time, paced_end_time, analysis_window_duration, TARGET_SAMPLE_RATE)  

print(f"Optimal Resting Window: {optimal_resting_start_idx/TARGET_SAMPLE_RATE:.1f}s to {optimal_resting_end_idx/TARGET_SAMPLE_RATE:.1f}s")
print(f"Optimal Paced Window: {optimal_paced_start_idx/TARGET_SAMPLE_RATE:.1f}s to {optimal_paced_end_idx/TARGET_SAMPLE_RATE:.1f}s")
