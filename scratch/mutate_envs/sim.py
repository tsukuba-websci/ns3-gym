#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

# @param [env, agent] EA
# @return integer スループット
# TODO: 全体的にリファクタ.ひとまず動く状態

def simulate(EA):
    environment = EA[0]
    agent = EA[1]
    startSim = True
    iterationNum = 1

    port = 5555
    simTime = 5  # seconds
    stepTime = 0.2  # seconds
    seed = 12
    simArgs = {"--duration": simTime,
               "--nLeaf": environment['n_leaf'],
               "--error_p": environment['error_rate'],
               "--bottleneck_bandwidth": environment['bottleneck_bandwidth'],
               "--bottleneck_delay": environment['bottleneck_delay'],
               "--access_bandwidth": environment['access_bandwidth'],
               "--access_delay": environment['access_delay'],
               "cross_traffic_data_rate": environment['cross_traffic_data_rate']
               }
    debug = False

    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                        simSeed=seed, simArgs=simArgs, debug=debug)
    # simpler:
    # env = ns3env.Ns3Env()
    env.reset()

    ob_space = env.observation_space
    ac_space = env.action_space
    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.dtype)

    stepIdx = 0
    currIt = 0

    # initialize variable
    get_agent.tcpAgents = {}
    get_agent.ob_space = ob_space
    get_agent.ac_space = ac_space

    try:
        while True:
            print("Start iteration: ", currIt)
            obs = env.reset()
            reward = 0
            done = False
            info = None
            # print("Step: ", stepIdx)
            # print("---obs: ", obs)

            # get existing agent of create new TCP agent if needed
            tcpAgent = get_agent(obs, agent)

            while True:
                stepIdx += 1
                action = tcpAgent.get_action(obs, reward, done, info)
                # print("---action: ", action)

                # print("Step: ", stepIdx)
                obs, reward, done, info = env.step(action)
                # print("---obs, reward, done, info: ", obs, reward, done, info)

                # get existing agent of create new TCP agent if needed
                tcpAgent = get_agent(obs, agent)

                if done:
                    stepIdx = 0
                    if currIt + 1 < iterationNum:
                        env.reset()
                    break

            currIt += 1
            if currIt == iterationNum:
                break

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        env.close()
        return obs


def get_agent(obs , agent):
    socketUuid = obs[0]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        tcpAgent = agent
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


if __name__ == "__main__":
    simulate()
