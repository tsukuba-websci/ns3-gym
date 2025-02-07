from env_reproduce import env_reproduce
from eligible_to_reproduce import eligible_to_reproduce
from mc_satisfied import mc_satisfied
# @param List[env, agent] EA_list environmentとagentのリスト
# @return List[env, agent]


def mutate_env(EA_list):
    parent_list = []

    for EA in EA_list:
        if eligible_to_reproduce(EA):
            parent_list.append(EA)

    # ENV_REPRODUCEはmutationする関数.
    # memo: 現在は親と同じ数の子を作成するようにしている.表の高い親のみを重点的に複製するなどのやり方もありそう
    child_list = env_reproduce(parent_list)

    child_list = mc_satisfied(child_list)

    EA_list.append(child_list)

    return EA_list
