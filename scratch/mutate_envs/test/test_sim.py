from sim import simulate
def test_env_reproduce():
    env = {
        'n_leaf': 3,
        'error_rate': 0.3,
        'bottleneck_bandwidth': '8Mbps',
        'bottleneck_delay': '30ms',
    }

    agent = 'dummy_agent'

    EA = [env, agent]

    assert simulate()
