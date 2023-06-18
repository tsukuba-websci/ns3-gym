### Function ###
def SlowStart_CongestionAvoidance_Algorithm(ssThresh: int, cWnd: int, segmentSize: int, segmentsAcked: int, bytesInFlight: int):
    ''' Return new_cWnd and new_ssThresh '''
    # Be careful not to increase the value of new_cWnd and new_ssThresh too much.
    
    # Slow Start
    if cWnd < ssThresh:
        new_cWnd = cWnd + segmentSize * segmentsAcked
        new_ssThresh = ssThresh
    # Congestion Avoidance
    else:
        new_cWnd = cWnd + (segmentSize * segmentsAcked * segmentSize) / (2 * bytesInFlight)
        new_ssThresh = cWnd / 2
    
    return new_cWnd, new_ssThresh
### Function ###
def FastRetransmit_FastRecovery_Algorithm(ssThresh: int, cWnd: int, segmentSize: int, segmentsAcked: int, bytesInFlight: int):
    ''' Return new_cWnd and new_ssThresh '''
    # Be careful not to increase the value of new_cWnd and new_ssThresh too much.
    
    # Fast Retransmit
    if segmentsAcked == 3:
        new_cWnd = cWnd // 2
        new_ssThresh = cWnd // 2
    
    # Fast Recovery
    elif bytesInFlight > 0:
        new_cWnd = cWnd + segmentSize
        new_ssThresh = cWnd
    
    # Congestion Avoidance
    else:
        new_cWnd = cWnd + (segmentSize * segmentsAcked)
        new_ssThresh = ssThresh
    
    return new_cWnd, new_ssThresh