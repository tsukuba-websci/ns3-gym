#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random
import time
from ns3gym import ns3env
from tcp_newreno import TcpNewReno
import cca

import openai
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import map_elites

from loky import get_reusable_executor

data_dir_path = 'data/'
code_dir_path = 'codes/'
result_dir_path = 'result_graph/'
os.makedirs(data_dir_path, exist_ok=True)
os.makedirs(code_dir_path, exist_ok=True)
os.makedirs(result_dir_path, exist_ok=True)



# GPT3 ===
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt3(prompt: str) -> str:
    p = 0.05
    t = 0.7  # 1.4?
    max_length = 256
    engine = "text-davinci-003"
    
    response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_length, temperature=t, top_p=p, echo=False)
    
    return response['choices'][0]['text']

def gpt35(prompt: str) -> str:
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

# feedback prompt
prompt_fb = make_prompt("feedback_prompt")

# improved function head
new_sc_head = make_prompt("new_sc_head")
new_ff_head = make_prompt("new_ff_head")

# small change prompt
prompt_small = """# There is one bug in this function. Please find and fix it."""


# ns3gym ===
def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            tcpAgent = cca.Dummy_cca()
        else:
            tcpAgent = TcpNewReno()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


def simulation(agent_sc, agent_ff, port):
    startSim = 1  # bool
    iterationNum = 1  # int

    simTime = 30 # seconds
    stepTime = 0.1  # seconds
    n_leaf = 2
    error_rate = 0.0
    bottleneck_bandwidth = "10Mbps"
    bottleneck_delay = "45ms"
    access_bandwidth = "10Mbps"
    access_delay = "45ms"
    cross_traffic_data_rate = "6Mbps"
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
            print("Step: ", stepIdx)
            print("---obs: ", obs)

            # obs[1] = 0    
            tcpAgent = get_agent(obs)
        
            # cWnd size
            cwnd = []
            cwnd.append(obs[5])

            while True:
                stepIdx += 1   
                    
                # dummy
                action = tcpAgent.get_action(obs, agent_sc, agent_ff, reward, done, info)
                obs, reward, done, info = env.step(action)
                # print(obs)
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
        
        
    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    
    env.close()    
    print("Done")
    
    return fitness, cwnd


def make_method(y_sc, y_ff):
    tail1 = "\nsc_method = SlowStart_CongestionAvoidance_Algorithm"
    tail2 = "\nff_method = FastRetransmit_FastRecovery_Algorithm" 
    
    return [y_sc + tail1, y_ff + tail2]

# for except
def f_sc(ssThresh: int, cWnd: int, segmentSize: int, segmentsAcked: int, bytesInFlight: int):
    new_cWnd = 1
    new_ssThresh = 1
    
    return new_cWnd, new_ssThresh

def f_ff(ssThresh: int, cWnd: int, segmentSize: int, segmentsAcked: int, bytesInFlight: int):
    new_cWnd = 1
    new_ssThresh = 1
    
    return new_cWnd, new_ssThresh    




# Mutation ===
def mutate(BD_list, sc_head, prompt_sc, history_sc, ff_head, prompt_ff, history_ff):    
    # Randomly select a niche
    bd_random = random.choice(BD_list)
    print("Mutate : (" + str(bd_random[0]) + ", " + str(bd_random[1]) + ")")
    
    # Read file and set cca
    with open(code_dir_path + 'cca_' + str(bd_random[0]) + '_' + str(bd_random[1]) + '/sc.py') as f:
        code_sc = f.read()
            
    with open(code_dir_path + 'cca_' + str(bd_random[0]) + '_' + str(bd_random[1]) + '/ff.py') as f:
        code_ff = f.read()

    # sc or ff
    if random.random() < 0.5:
        # muate sc
        p = random.random()
                    
        # change function
        if p < 0.2:
            code_sc = sc_head + gpt3(prompt_sc) 
                    
        # self-refine
        elif p < 0.6:
            for _ in range(2):
                # feadback
                prompt_im = gpt3(code_sc + prompt_fb)
                
                # history
                prompt_history = """### Examples ###\n"""
                for t in history_sc.keys():
                    prompt_history += t
                
                # improve
                code_sc = sc_head + gpt3(prompt_history + prompt_im + new_sc_head)                
                    
        # small change
        else:
            code_sc = sc_head + gpt3(prompt_small + code_sc + new_sc_head)
                
    else:
        # mutate ff
        p = random.random()
        
        # change function
        if p < 0.2:
            code_ff = ff_head + gpt3(prompt_ff)
        
        # self-refine
        elif p < 0.6:
            for _ in range(2):
                # feadback
                prompt_im = gpt3(code_ff + prompt_fb)
                
                # history
                prompt_history = """### Examples ###\n"""
                for t in history_ff.keys():
                    prompt_history += t
                
                # improve
                code_ff = ff_head + gpt3(prompt_history + prompt_im + new_ff_head)
                
        # small change
        else:
            code_ff = ff_head + gpt3(prompt_small + code_ff + new_ff_head)     
            
    return [code_sc, code_ff]


def sort_value(d):
    sort_list = sorted(d.items(), key = lambda c : c[1], reverse=True)
    return dict((x, y) for x, y in sort_list)



# main ===
if __name__ == "__main__":
    # MAP-Elites ===
    bin = 50
    range_bd1 = [0.0, 3.0e5]
    range_bd2 = [0.0, 1.5e5]
    N = 3  # 48 (population size)
    max_iteration = 3  # 5600 
    
    # dummy
    sc_method = f_sc
    ff_method = f_ff
    
    # main loop
    for iter in range(max_iteration):
        print("\nIteration :", iter)
        
        # iteration < 1
        if (iter < 1):
            # init map_elites
            index_bd1 = np.round(np.linspace(range_bd1[0], range_bd1[1], bin), 2)
            index_bd2 = np.round(np.linspace(range_bd2[0], range_bd2[1], bin), 2)

            make_archive = np.empty((bin, bin))
            make_archive[:] = None
            
            archive = pd.DataFrame(make_archive, index=index_bd2, columns=index_bd1)
            BD_list = []
            file_info = [False, [0.0, 0.0]]
            
            # generate init cca
            prompt_sc = make_prompt("prompt_sc")
            sc_head = make_prompt("sc_head")
            prompt_sc += sc_head
            init_sc = sc_head + gpt3(prompt_sc)
            
            prompt_ff = make_prompt("prompt_ff")
            ff_head = make_prompt("ff_head")
            prompt_ff += ff_head
            init_ff = ff_head + gpt3(prompt_ff)
            
            # make_method
            init_sc_method, init_ff_method = make_method(init_sc, init_ff)
            try:
                exec(init_sc_method)
                exec(init_ff_method)
                sc = sc_method
                ff = ff_method

            except:
                sc = f_sc
                ff = f_ff
            
            # evaluate
            fitness, cwnd = simulation(sc, ff, 5050)
            
            # update BD_list and archive
            BD_list, archive, file_info = map_elites.update_map(cwnd, fitness, BD_list, archive, index_bd1, index_bd2)
                    
            # save cca
            os.makedirs(code_dir_path + 'cca_' + str(BD_list[0][0]) + '_' + str(BD_list[0][1]), exist_ok=True)
            with open(code_dir_path + 'cca_' + str(BD_list[0][0]) + '_' + str(BD_list[0][1]) + '/sc.py', 'w') as f:
                f.write(init_sc)
                
            with open(code_dir_path + 'cca_' + str(BD_list[0][0]) + '_' + str(BD_list[0][1]) + '/ff.py', 'w') as f:
                f.write(init_ff)
                
            # init history
            code_sc = init_sc
            code_ff = init_ff
            history_sc = {code_sc : fitness}
            history_ff = {code_ff : fitness}
            
            
        # itreration >= 1
        else:
            # agent list
            mutated_outputs = []
            cca_method = []

            '''
            # Parallel processing
            executor1 = get_reusable_executor()
            for i in range(N):   
                mutated_output = executor1.submit(mutate, BD_list, sc_head, prompt_sc, history_sc, ff_head, prompt_ff, history_ff)
                mutated_outputs.append(mutated_output)  # append cca_sc and cca_ff (str)
            '''
            
            # Not Parallel
            for i in range(N):
                mutated_outputs.append(mutate(BD_list, sc_head, prompt_sc, history_sc, ff_head, prompt_ff, history_ff))
                
            # from str to method
            for i in range(N):
                '''
                # Parallel processing
                cca_method.append(make_method(mutated_outputs[i].result()[0], mutated_outputs[i].result()[1]))   
                '''     
                
                # Not Parallel     
                cca_method.append(make_method(mutated_outputs[i][0], mutated_outputs[i][1]))
                
                try:
                    exec(cca_method[i][0])
                    exec(cca_method[i][1])
                    cca_method[i][0] = sc_method
                    cca_method[i][1] = ff_method
                except:
                    cca_method[i][0] = f_sc
                    cca_method[i][1] = f_ff
            
            # evaluate (ns3)
            # Parallel processing
            simulated_outputs = []
            executor2 = get_reusable_executor()
            for i in range(N):
                simulated_output = executor2.submit(simulation, cca_method[i][0], cca_method[i][1], 8888+i)
                simulated_outputs.append(simulated_output)  # append fitness and cwnd
            
            # update history
            for i in range(N):
                history_sc[mutated_outputs[i][0]] = simulated_outputs[i].result()[0]
                history_ff[mutated_outputs[i][1]] = simulated_outputs[i].result()[0]
            
            # sort    
            history_sc = sort_value(history_sc)
            history_ff = sort_value(history_ff)
            
            # select top 5
            history_sc = dict(list(history_sc.items())[:5])
            history_ff = dict(list(history_ff.items())[:5])
            
            # update BD_list and archive 
            for i in range(N):
                BD_list, archive, file_info = map_elites.update_map(simulated_outputs[i].result()[1], simulated_outputs[i].result()[0], BD_list, archive, index_bd1, index_bd2)
                
                if (file_info[0]):
                    os.makedirs(code_dir_path + 'cca_' + str(file_info[1][0]) + '_' + str(file_info[1][1]), exist_ok=True)
                    with open(code_dir_path + 'cca_' + str(file_info[1][0]) + '_' + str(file_info[1][1]) + '/sc.py', 'w') as f:
                        f.write(mutated_outputs[i][0])
                    
                    with open(code_dir_path + 'cca_' + str(file_info[1][0]) + '_' + str(file_info[1][1]) + '/ff.py', 'w') as f:
                        f.write(mutated_outputs[i][1])

    # archive map
    archive = pd.DataFrame(archive, index=index_bd2[::-1], columns=index_bd1)

    # save csv
    archive.to_csv(data_dir_path + 'archive_map.csv')
    # plot
    archive = archive.rename(columns=lambda c: int(c), index=lambda i: int(i))
    
    # plot
    archive = archive.rename(columns=lambda c: int(c), index=lambda i: int(i))
    sns.heatmap(archive, mask=(archive==None), xticklabels=10, yticklabels=10)
    plt.xlabel("mean(congestion window size)")
    plt.ylabel("std(congestion window size)")
   
    # save image
    plt.savefig(result_dir_path + "archive_map.png")
