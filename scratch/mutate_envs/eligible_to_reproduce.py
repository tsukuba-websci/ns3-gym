# @param [env, agent] EA
# @return [env, agent]
from sim import simulate

def eligible_to_reproduce(EA):
    condition = 10000
    obs = simulate(EA)
    if obs[5] > condition:
        return True
    else:
        return False