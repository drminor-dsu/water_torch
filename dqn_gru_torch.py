#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:44:13 2024

@author: drminor
"""


import torch
from torchviz import make_dot
from water_gym import WaterGym, Actions
from dqn_model import DqnGRU

import re
import numpy as np
import random
import copy
from collections import deque
from matplotlib import pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actions = Actions
nstates = 5 # how to many state data use (the number of previous states: 2 + current state: 1)
num_layers = 3 # the number of layers of GRU model
hidden_dim = 32 # ther number of nodes for a layer
minutes = 2 # the interval minutes for simulation


def stratified_split_data(years, durations, trate=0.8, vrate=0.0, stratify=True):
    """
    /home/drminor/projects/water/torch/data/gasan/10year/10yr_0010m_h055.inp
    year_freq=[10, 20, 30, 50, 80, 100],
    duration=[0010, 0060, 0120, 0180, 0240, 0360, 0540, 0720, 1080, 1440]
    
    Parameters
    ----------
    years : string list
        years = ['10', '20', '30', '50', '80', '100'],
    minutes : string list
        minutes = ['0010', '0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']

    Returns
    -------
    inp_list : TYPE
        DESCRIPTION.
    train_inps : TYPE
        DESCRIPTION.
    valid_inps : TYPE
        DESCRIPTION.
    test_inps : TYPE
        DESCRIPTION.

    """
    with open('../data/selected_inp.txt', 'r') as fd:
        inp_list = fd.readlines()
        inp_list = [f.strip() for f in inp_list] 
        inp_list = [f for f in inp_list if (re.findall(r'\d+', f.split('/')[-1])[0] in years)  
                    and (re.findall(r'\d+', f.split('/')[-1])[1] in durations)]
    
    inp_list.sort()
    train_inps = []
    valid_inps = []
    test_inps = []
    if stratify:
        for y in years:
            for m in durations:
                data = [f for f in inp_list 
                            if re.findall(r'\d+', f.split('/')[-1])[0] == y
                            and re.findall(r'\d+', f.split('/')[-1])[1] == m]
                random.shuffle(data)
                nsample = len(data) # the number of total sample
                ntrate = int(nsample * trate) # the number of training sample
                nvrate = int(nsample * vrate) # the number of validation sample
                
                train_inps.extend(data[:ntrate])
                valid_inps.extend(data[ntrate:ntrate+nvrate])
                test_inps.extend(data[ntrate+nvrate:])
    random.shuffle(train_inps)
    random.shuffle(valid_inps)
    # random.shuffle(test_inps)
    
    return inp_list, train_inps, valid_inps, test_inps


def data_split(trate:float=0.7, vrate:float=0.1):
    """
    Data splition with training data rate, validation data rate, test data rate
    test data rate = 1.0 - (training data rate + validation data rate)
    
    Parameters
    ----------
    trate : TYPE, optional
        Training data rate. The default is 0.7.
    vrate : TYPE, optional
        Validation data rate. The default is 0.1.

    Returns
    -------
    train_inps : list
        train data set
    valid_inps : list
        validation data set
    test_inps : list
        test data set

    """
    with open('../data/selected_inp.txt', 'r') as fd: # 마이너스 강우량이 보정된 inp 파일 리스트 데이터
        inp_list = fd.readlines()
        inp_list = [f.strip() for f in inp_list]
    
    random.shuffle(inp_list)
    nsample = len(inp_list) # the number of total sample
    ntrate = int(nsample * trate) # the number of training sample
    nvrate = int(nsample * vrate) # the number of validation sample
    
    train_inps = inp_list[:ntrate]
    valid_inps = inp_list[ntrate:ntrate+nvrate]
    test_inps = inp_list[ntrate+nvrate:]
    
    return inp_list, train_inps, valid_inps, test_inps
    

def train(train_inps, valid_inps, epochs:int=1, minutes:int=1):
    """
    
    
    Parameters
    ----------
    train_inps: TYPE
        Training data rate. The default is 0.7.
    valid_inps : TYPE, optional
        Validation data rate. The default is 0.1.
    minutes : int, optional
        펌프 조합 의사결정 시간단위. 기본 값 1분
    
    Returns
    -------
    
    """

    train_inps = copy.deepcopy(train_inps)    
    policy_model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                          output_dim=len(actions), 
                          num_layers=num_layers).to(device)
    
    target_model = copy.deepcopy(policy_model).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    
    gamma = 0.1
    
    epsilon = 1.0
    train_epoch_losses = []
    valid_epoch_losses = []
    mem_size = 100
    batch_size = 20
    replay = deque(maxlen=mem_size)
    sync_freq = 10 #A
    
    for epoch in range(epochs):
        print(f'Epoch # {epoch}/{epochs}')
        num_samples = len(train_inps)
        losses = []
        for i in range(num_samples): 
            print(f'\r{i+1}/{num_samples}', end='', flush=True)
            gym = WaterGym(train_inps[i], minutes)
            state_, reward, done, info = gym.reset()
            
            input_q = deque([], nstates) # for build input vector
            # build intial input 
            for k in range(nstates-1): 
                input_q.append(np.array([0.0 for _ in range(len(state_))]))
            state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
            input_q.append(state_)
            state = torch.tensor(np.stack(input_q), dtype=torch.float32, device=device)
            
            j = 0
            status = 1
            while(status): 
                j+=1
                state_prev = torch.Tensor.cpu(state)
                q_val, hidden_state = policy_model(state.unsqueeze(0))
                q_val_ = torch.Tensor.cpu(q_val).data.numpy()
                if (random.random() < epsilon):
                    action = np.random.randint(0,len(actions))
                else:
                    action = np.argmax(q_val_)
                #print(action)
                
                state_, reward, done, info = gym.step(action)
                state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
                input_q.append(state_)
                state = torch.tensor(np.stack(input_q),
                                     dtype=torch.float32, device=device)
                
                exp =  (state_prev, action, reward, torch.Tensor.cpu(state), done)
                replay.append(exp) #H
                
                if len(replay) > batch_size:
                    minibatch = random.sample(replay, batch_size)
                    state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                    action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                    reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                    state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                    done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)
                       
                    Q1_gru, hidden_state = policy_model(state1_batch) 
                    with torch.no_grad():
                        Q2_gru, hidden_state = target_model(state2_batch)
                    
                    Y = reward_batch + gamma * (1-done_batch) * torch.max(Q2_gru, dim=1)[0]
                    X = Q1_gru.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())
                    #print(i, loss.item())
                 
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    
                if j % sync_freq == 0:
                    target_model.load_state_dict(policy_model.state_dict())
                if done == True:
                    status = 0
            if epsilon > 0.2:
                epsilon -= (1/num_samples)
        train_epoch_losses.append(np.asarray(losses).mean())
    print() # for output line adjustment                 
    
    train_epoch_losses = np.array(train_epoch_losses)
    
    torch.save(policy_model.state_dict(), '../trained_models/water_dqn_gru.pkl')
    
    return policy_model, train_epoch_losses


def plot_losses(losses):
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Loss",fontsize=22)


"""
state_, reward, done, info , <= 
        [self.rains[self.clock], 
        cur_inflow, 
        cur_outflow, 
        cur_vol, 
        cur_level],\
        cur_reward, done, info
        
info <= (vol_reward, act_reward, energy_reward, excess_pump * excess_pump_penalty)
"""

def test_model(model, test_inp, minutes:int=1):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model == None:
        model = DqnGRU(input_dim=5, hidden_dim=hidden_dim, 
                       output_dim=len(actions), num_layers=num_layers).to(device)
        model.load_state_dict(torch.load('../trained_models/water_dqn_gru.pkl'))
        model.eval()
    
    gym = WaterGym(test_inp, minutes)
    init_state_, reward, done, info = gym.reset()
    input_q = deque([], nstates) # for build input vector
    for k in range(nstates-1): 
        input_q.append(np.array([0.0 for _ in range(len(init_state_))]))
    state_ = np.array(init_state_) / (np.array(init_state_).sum() + 0.00001) # input normalization
    input_q.append(state_)
    state = torch.tensor(np.stack(input_q),
                         dtype=torch.float32, 
                         device=device)
    
    action_list = [0]
    state_list = [init_state_]
    reward_list = [reward]
    info_list = [info]
    while(1):
        #print(state.unsqueeze(0).unsqueeze(0).shape)
        q_val, hidden_state = model(state.unsqueeze(0))
        q_val_ = torch.Tensor.cpu(q_val).data.numpy()
        #print(q_val_)
        action = np.argmax(q_val_)
        action_list.append(action)
        
        state_, reward, done, info = gym.step(action)
        
        # for performance evaluations
        state_list.append(state_) 
        reward_list.append(reward)
        info_list.append(info)
        
        state_ = np.array(state_) / (np.array(state_).sum() + 0.00001) # input normalization
        input_q.append(state_)
        state = torch.tensor(np.stack(input_q),
                             dtype=torch.float32,
                             device=device)
        
        if done == True:
            break
        
    return model, action_list, state_list, reward_list, info_list


def plot_performance(state_list, actions, 
                     kind=['rain', 'inflow', 'outflow', 'vol', 'level']):
    rain_list = []
    inflow_list = []
    outflow_list = []
    vol_list = []
    level_list = []
    
    
    for state in state_list:
        rain, inflow, outflow, vol, level = state
        rain_list.append(rain)
        inflow_list.append(inflow)
        outflow_list.append(outflow)
        vol_list.append(vol)
        level_list.append(level)
    
    kind_dic = {'rain': rain_list, 'inflow': inflow_list, 
                'outflow': outflow_list, 'vol': vol_list, 'level': level_list}
    color = ['black', 'blue', 'red', 'green', 'yellow']
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    for i in range(len(kind)-1):
        ax[0].plot(range(len(kind_dic[kind[i]])), 
                kind_dic[kind[i]], 
                color=color[i],
                label=list(kind_dic.keys())[i])
    ax[0].legend()
    
    ax[1].plot(range(len(actions)),
                 kind_dic['rain'],
                 color=color[0],
                 label='rain/m')
    ax[1].plot(range(len(actions)),
                 kind_dic['level'],
                 color=color[1],
                 label='level')
    ax[1].scatter(range(len(actions)),
                 actions,
                 color=color[2],
                 label='actions')
    ax[1].legend()
    
    fig.show()
        

def plot_individual_case(state_list, action_list):
    """
    [self.rains[self.clock], 
    cur_inflow, 
    cur_outflow, 
    cur_vol, 
    cur_level]

    Parameters
    ----------
    state_list : TYPE
        DESCRIPTION.
    action_list : TYPE
        DESCRIPTION.
    ['rain', 'inflow', 'outflow', 'vol', 'level'] : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rain = []
    inflow = []
    outflow = []
    volume = []
    level = []
    
    for state in state_list:
        rain.append(state[0])
        inflow.append(state[1])
        outflow.append(state[2])
        volume.append(state[3])
        level.append(state[4])
    
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(1,2, sharex=False)
    fig.set_size_inches(30, 23)
    
    label_size = 25
    tick_size = 20
    legend_size = 25
    
    ax[0].set_xlabel('Time($min$)', fontsize=label_size, fontweight='bold')
    ax[0].set_ylabel('Volume($m^3$)', fontsize=label_size, fontweight='bold')
    ax[0].plot(inflow, color='r', marker='D', label= 'inflow into the reservoir')
    ax[0].plot(outflow, color='b', marker='o', label= 'outflow from the reservoir')
    ax[0].tick_params(labelsize=tick_size)
    ax[0].legend(loc='best', fontsize=legend_size)
    
    ax[1].set_xlabel('Time($min$)', fontsize=label_size, fontweight='bold')
    ax[1].set_ylabel('Elevation($m$)', fontsize=label_size, fontweight='bold')
    ax[1].plot(level, color='b', marker='o', label= 'reservoir elevation')
    ax[1].legend(loc='best', fontsize=legend_size)
    
    plt.tight_layout()
    plt.show()


def plot_reward(reward_list, info_list):
    pass
    

if __name__ == '__main__':
    
    years = ['10', '20', '30', '50', '80', '100']
    durations = ['0010', '0060', '0120', '0180', '0240', '0360', '0540', '0720', '1080', '1440']
    minutes = 2  
    epochs = 1

    #total, train_list, valid_list, test_list = data_split(trate=0.1)
    inp, train_inps, valid_inps, test_inps = stratified_split_data(years, durations, trate=0.1, vrate=0.1, stratify=True)
    #model, losses = train(train_inps, valid_inps, epochs=epochs, minutes=minutes)
    #plot_losses(losses)
    test_list = ['/home/drminor/projects/water/torch/data/gasan/30year/30yr_0720m_h954.inp']
    tmodel, action_list, state_list, reward_list, info_list = test_model(None, test_list[0], minutes=minutes)
    plt.figure()
    plot_individual_case(state_list, action_list)