from tcp_newreno import TcpNewReno
from sim import simulate


def test_sim():
    env = {
        'n_leaf': 3,
        'error_rate': 0.3,
        'bottleneck_bandwidth': '8Mbps',
        'bottleneck_delay': '30ms',
        'access_bandwidth': "10Mbps",
        'access_delay': "45ms",
        'cross_traffic_data_rate': "6Mbps"
    }

    agent = TcpNewReno()

    EA = [env, agent]
    obs = simulate(EA)
# obs[5]はcwnd
    assert bool(obs[5]) == True