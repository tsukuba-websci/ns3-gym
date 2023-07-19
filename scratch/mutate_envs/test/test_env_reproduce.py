from env_reproduce import env_reproduce
import copy
def test_env_reproduce():
    env_1 = {
        'n_leaf': 3,
        'error_rate': 0.3,
        'bottleneck_bandwidth': '8Mbps',
        'bottleneck_delay': '30ms',
    }
    env_2 = {
        'n_leaf': 6,
        'error_rate': 0.7,
        'bottleneck_bandwidth': '15Mbps',
        'bottleneck_delay': '50ms',
    }

    agent = 'agent'

    EA = [[env_1, agent], [env_2, agent]]
    # 参照渡しだと、EAの値が変わってしまう。また、二次元配列なのでdeepcopy()する必要あり
    EA_copy = copy.deepcopy(EA)
    mutated_EA = env_reproduce(EA_copy)

    assert EA[0][0] is not mutated_EA[0][0]
    assert EA[1][0] is not mutated_EA[1][0]
