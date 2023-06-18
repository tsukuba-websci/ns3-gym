import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# BD
def search_bd(agent, index_bd):
    if agent >= index_bd[-1]:
        bd = index_bd[-1]
    else:
        for i in index_bd:
            if agent < i:
                bd = i
                break
    return bd


def update_map(result_list, fitness, BD_list, archive, index_bd1, index_bd2):
    # evaluate bd
    mean_cwnd, std_cwnd = np.mean(result_list), np.std(result_list)
    
    # niche
    bd1 = search_bd(mean_cwnd, index_bd1)
    bd2 = search_bd(std_cwnd, index_bd2)
    
    # Determine whether to record
    if (np.isnan(archive[bd1][bd2])) or (archive[bd1][bd2] < fitness):
        archive[bd1][bd2] = fitness
        
        # update BD_list based on archive 
        if (not ([bd1, bd2] in BD_list)):
            BD_list.append([bd1, bd2])
        
        # update file_info based on archive 
        file_info = [True, [bd1, bd2]]
        
    else:
        file_info = [False, [bd1, bd2]]    
    
    return BD_list, archive, file_info
