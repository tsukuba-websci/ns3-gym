# @param List[env, agent] EA_list
# @return List[env, agent]
from sim import simulate

def mc_satisfied(EA_list):
    passed_EA_list = []
    condition = 10000
    for EA in EA_list:
        obs = simulate(EA)
        if obs[5] > condition:
            passed_EA_list.append(EA)
    
    return passed_EA_list