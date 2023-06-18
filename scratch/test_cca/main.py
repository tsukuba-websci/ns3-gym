import os
import sys
import argparse
import openai
from ns3gym import ns3env
from tcp_newreno import TcpNewReno
import my_cca

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

openai.api_key = os.getenv("OPENAI_API_KEY")

data_dir_path = 'data/'
code_dir_path = 'codes/'
result_dir_path = 'result_graph/'
os.makedirs(data_dir_path, exist_ok=True)
os.makedirs(code_dir_path, exist_ok=True)
os.makedirs(result_dir_path, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt35_codex(prompt: str) -> str:
    p = 0.05
    t = 0.7
    max_length = 256
    engine = "text-davinci-003"
    
    response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_length, temperature=t, top_p=p, echo=False)
    
    return response['choices'][0]['text']


def gpt35_turbo(prompt: str) -> str:
    model = "gpt-3.5-turbo"
    messages = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
        )
    
    return response["choices"][0]["message"]["content"]


def make_prompt(prompt_path: str) -> str:
    with open("prompts/" + prompt_path + ".txt", "r") as f:
        prompt = f.read()
        
    return prompt


def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            tcpAgent = my_cca.MyCCA()
        else:
            tcpAgent = TcpNewReno()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


def simulation(port, simTime = 30, stepTime = 0.1, n_leaf = 2, error_rate = 0.0, bottleneck_bandwidth = "10Mbps", bottleneck_delay = "45ms", access_bandwidth = "10Mbps", access_delay = "45ms", cross_traffic_data_rate = "6Mbps"):
    startSim = 1  # bool
    iterationNum = 1  # int

    seed = 12
    simArgs = {"--duration": simTime,
               "--nLeaf": n_leaf,
               "--error_p": error_rate,
               "--bottleneck_bandwidth": bottleneck_bandwidth,
               "--bottleneck_delay": bottleneck_delay,
               "--access_bandwidth": access_bandwidth,
               "--access_delay": access_delay,
               "cross_traffic_data_rate": cross_traffic_data_rate
               }
    debug = False

    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    env.reset()

    ob_space = env.observation_space
    ac_space = env.action_space
    print("Observation space: ", ob_space, ob_space.dtype)
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
            print("Step: ", stepIdx)
            print("---obs: ", obs)

            # get existing agent of create new TCP agent if needed
            my_cca.MyCCA.update_cca()  # reload
            tcpAgent = get_agent(obs)
        
            # plot
            cwnd = [obs[5]]

            while True:
                stepIdx += 1
                action = tcpAgent.get_action(obs, reward, done, info)
                obs, reward, done, info = env.step(action)
                print("---obs: ", obs)
                
                # plot
                cwnd.append(obs[5])

                if done:
                    print("Step:", stepIdx)
                    print("obs:", obs)
                    throughput = pd.read_csv(data_dir_path + "/throughput" + str(port) + ".csv", header=None)
                    fitness = throughput.mean()[1]
                    print("mean throughput:", fitness)
                
                    cwnd = np.array(cwnd)
                    print("mean cwnd:", np.mean(cwnd))
                    print("std cwnd:", np.std(cwnd))
                    stepIdx = 0
                    if currIt + 1 < iterationNum:
                        env.reset()
                    break

            currIt += 1
            if currIt == iterationNum:
                break
        
        # plot
        t = np.arange(0, simTime, simTime/len(cwnd))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Congestion Window (pkts)")
        ax.plot(t, cwnd, label="mean(throughput) = " + str(round(fitness)) + "kbps")
        ax.legend()
        fig.savefig(result_dir_path + "test_cca.png")
        
    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    
    env.close()    
    print("Done")
    
    return fitness, cwnd



def main():    
    # slow start and congestion avoidance
    prompt_sc = make_prompt("prompt_sc")
    sc_head = make_prompt("sc_head")
    prompt_sc += sc_head
    init_sc = sc_head + gpt35_codex(prompt_sc)
    
    # fast transmit and fast recovary
    prompt_ff = make_prompt("prompt_ff")
    ff_head = make_prompt("ff_head")
    prompt_ff += ff_head
    init_ff = ff_head + gpt35_codex(prompt_ff)
    
    generated_cca = init_sc + "\n" + init_ff
    
    with open(code_dir_path + "generated_cca.py", "w") as f:
        f.write(generated_cca)
    
    # ns3-gym  
    simulation(5555)
    

if __name__ == "__main__":
    main()