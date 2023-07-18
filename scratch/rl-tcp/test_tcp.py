#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="output.log",
    filemode="w",
)

parser = argparse.ArgumentParser(description="Start simulation script on/off")
parser.add_argument(
    "--start", type=int, default=1, help="Start ns-3 simulation script 0/1, Default: 1"
)
parser.add_argument(
    "--iterations", type=int, default=1, help="Number of iterations, Default: 1"
)
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10  # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {
    "--duration": simTime,
}
debug = False

env = ns3env.Ns3Env(
    port=port,
    stepTime=stepTime,
    startSim=startSim,
    simSeed=seed,
    simArgs=simArgs,
    debug=debug,
)
# simpler:
# env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
logging.info("Observation space: %s, %s", ob_space, ob_space.dtype)
logging.info("Action space: %s, %s", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0


def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            # time-based = 1
            tcpAgent = TcpTimeBased()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

try:
    print("step,ssThresh,windowSize")
    while True:
        obs = env.reset()
        reward = 0
        done = False
        info = None

        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs)

        while True:
            stepIdx += 1
            action = tcpAgent.get_action(obs, reward, done, info)
            print(f"{stepIdx},{action[0]},{action[1]}")

            obs, reward, done, info = env.step(action)
            logging.info("Step: %s", stepIdx)
            logging.info(
                "obs: %s, reward: %s, done: %s, info: %s", obs, reward, done, info
            )

            # get existing agent of create new TCP agent if needed
            tcpAgent = get_agent(obs)

            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    logging.warn("Ctrl-C received, exiting...")
finally:
    env.close()
    logging.info("Simulation done.")
