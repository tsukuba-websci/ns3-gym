# @param List[env, agent] EA_list environmentとagentのリスト
# @return List[env, agent]

# memo: 現在は親と同じ数の子を作成するようにしている.表の高い親のみを重点的に複製するなどのやり方もありそう
from env_params import env_params
import random
from params_utl import to_number, to_string

def env_reproduce(EA_list):
    child_list = []
    for EA in EA_list:
        mutated_EA = mutate(EA)
        child_list.append(mutated_EA)

    return child_list

# 各プロパティをstep、最大値、最小値に基づきmutateさせる。
def mutate(EA):
    enviroments = EA[0]
    enviroments = to_number(enviroments)
    env_keys = list(enviroments.keys())
    agent = EA[1]

    for env in env_keys:
        random_num = random.randrange(1)

        if (random_num == 0):
            if (enviroments[env] - env_params[env]['mutation_step'] < env_params[env]['min_value']):
                break
            enviroments[env] -= env_params[env]['mutation_step']
        elif (random_num == 1):
            if (enviroments[env] + env_params[env]['mutation_step'] > env_params[env]['max_value']):
                break
            enviroments[env] += env_params[env]['mutation_step']
    enviroments = to_string(enviroments)

    return [enviroments, agent]
