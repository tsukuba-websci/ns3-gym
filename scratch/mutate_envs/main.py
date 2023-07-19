# @param List[env, agent] EA_list environmentとagentのリスト
# @param integer max_children_num 複製時に作られる子の最大数
# @return List[env, agent]
def main(EA_list, max_children_num):
    parent_list = []

    for EA in EA_list:
        parent_list.append(EA)

    # ENV_REPRODUCEはmutationする関数
    child_list = env_reproduce(parent_list, max_children_num)

    EA_list.append(child_list)

    return EA_list

if __name__ == "__main__":
    main()