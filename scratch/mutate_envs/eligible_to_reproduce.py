# @param [env, agent] EA
# @return [env, agent]
from sim import simulate

def eligible_to_reproduce(EA):
    condition = 10000
    result = simulate(EA)
    if result > condition:
        return True
    else:
        return False