from tcp_base import TcpEventBased

class Dummy_cca(TcpEventBased):
    def __init__(self):
        super(Dummy_cca, self).__init__()

    def get_action(self, obs, agent_sc, agent_ff, reward, done, info):
        try:
            new_cWnd, new_ssThresh = agent_sc(obs[4], obs[5], obs[6], obs[7], obs[8])
            new_cWnd, new_ssThresh = agent_ff(new_ssThresh, new_cWnd, obs[6], obs[7], obs[8])
        except:
            new_cWnd = 1
            new_ssThresh = 1
            
        # limit
        new_cWnd = new_cWnd if new_cWnd < 300000 else 300000
        new_cWnd = new_cWnd if new_cWnd > 1 else 1
        new_ssThresh = new_ssThresh if new_ssThresh < 300000 else 300000
        new_ssThresh = new_ssThresh if new_ssThresh > 1 else 1
        
        actions = [int(new_ssThresh), int(new_cWnd)]

        return actions
