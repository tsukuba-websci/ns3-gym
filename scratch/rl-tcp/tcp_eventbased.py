from tcp_base import TcpEventBased


class TcpEventBase(TcpEventBased):
    """docstring for TcpEventBased"""

    def __init__(self):
        super(TcpEventBased, self).__init__()

    def get_action(self, obs, reward, done, info):
        """docstring for get_action

        Args:
            obs (_type_): _description_
            reward (_type_): _description_
            done (function): _description_
            info (_type_): _description_

        Returns:
            _type_: _description_
        """
        # unique socket ID
        socketUuid = obs[0]
        # TCP env type: event-based = 0 / time-based = 1
        envType = obs[1]
        # sim time in us
        simTime_us = obs[2]
        # unique node ID
        nodeId = obs[3]
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # bytesInFlightSum
        bytesInFlightSum = obs[7]
        # bytesInFlightAvg
        bytesInFlightAvg = obs[8]

        # compute new values
        new_cWnd = 10 * segmentSize
        new_ssThresh = 5 * segmentSize

        # return actions
        actions = [new_ssThresh, new_cWnd]

        return actions
