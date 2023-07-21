from tcp_base import TcpEventBased
import generated_cca as cca
import importlib

class MyCCA(TcpEventBased):
    def __init__(self):
        super(MyCCA, self).__init__()
        
    def update_cca():
        importlib.reload(cca)

    def get_action(self, obs, reward, done, info):
        try:
            new_cWnd, new_ssThresh = cca.SlowStart_CongestionAvoidance_Algorithm(obs[4], obs[5], obs[6], obs[7], obs[8])
            new_cWnd, new_ssThresh = cca.FastRetransmit_FastRecovery_Algorithm(new_ssThresh, new_cWnd, obs[6], obs[7], obs[8])
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
