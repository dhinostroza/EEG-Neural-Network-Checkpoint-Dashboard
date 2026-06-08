import numpy as np

def apply_transition_rules(predictions):
    """
    Applies biological consistency rules to a sequence of sleep stage predictions.
    
    Args:
        predictions (list or np.array): Sequence of predicted stages (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
        
    Returns:
        np.array: Corrected predictions.
    """
    # Ensure numpy array
    corrected = np.array(predictions, dtype=int).copy()
    n = len(corrected)
    
    # --- Pass 1: Smoothing (Remove isolated blips first) ---
    # Removes 1-epoch anomalies like N2 -> W -> N2 (likely artifact)
    smoothed = corrected.copy()
    for i in range(1, n-1):
        prev_s = corrected[i-1]
        curr_s = corrected[i]
        next_s = corrected[i+1]
        
        # If neighbors are identical but center is different -> smoothen
        if prev_s == next_s and curr_s != prev_s:
            smoothed[i] = prev_s
            
    corrected = smoothed

    # --- Pass 2: Transition Rules (Biological Constraints) ---
    
    # Rule: Wake (0) -> REM (4) is biologically impossible/rare.
    # Logic: If we see W -> REM, the REM is likely N1 (the bridge stage).
    for i in range(1, n):
        if corrected[i-1] == 0 and corrected[i] == 4:
            corrected[i] = 1 # Correct REM to N1

    # Rule: N1 (1) -> REM (4) is also highly unlikely (except SOREMP).
    # Logic: If we see N1 -> REM, it's likely N1 (continuation) or N2.
    # Given model tendency to confuse N1/REM, we bias towards N1.
    for i in range(1, n):
        if corrected[i-1] == 1 and corrected[i] == 4:
            corrected[i] = 1 # Correct REM to N1
            
    return corrected
