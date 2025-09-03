import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import gym
from ruamel.yaml import YAML

from ml.models.reward import MLPReward
from common.dataset import RankingLimitDataset,rank_collate_func
from common.sac import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    
    device='cpu'
    #load 
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']
    
    #load RLHF yaml
    use_gpu=False
    num_epochs=v['RLHF']['epoch']   #20
    save_interval=v['RLHF']['save_interval']   #10
    log_interval=v['RLHF']['log_interval']   #100
    batch_size=v['RLHF']['batch_size']  #64
    output_model_path=v['RLHF']['output_model_path']
    env = gym.make(env_name)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]


    train_demo_files=['HalfCheetah-v3-exp.pt','HalfCheetah-v3-med.pt','HalfCheetah-v3-medexp.pt']
    train_dataset = RankingLimitDataset(train_demo_files, None, num_inputs, num_actions, mode='state_action', traj_len=50)
    test_dataset = RankingLimitDataset(train_demo_files, None, num_inputs, num_actions, mode='state_action', traj_len=50)
    
    train_loader = data_utils.DataLoader(train_dataset, collate_fn=rank_collate_func, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data_utils.DataLoader(test_dataset, collate_fn=rank_collate_func, batch_size=1, shuffle=True, num_workers=4)
    
    reward_net = MLPReward(num_inputs+num_actions, **v['reward'], device=device).to(device)
    optimizer = optim.Adam(reward_net.parameters(), lr=0.001, weight_decay=0.0005)
    
    best_acc=0
    for epoch in range(num_epochs):
        counter = 0
        acc_counter = 0
        if epoch % save_interval == 0:
            for iter_, data in enumerate(test_loader):
                traj1, rew1, traj2, rew2 = data
                if use_gpu:
                    traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
                bs1 = len(traj1)
                bs2 = len(traj2)
                assert bs1 == bs2
                pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
                pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
                pred_rank = torch.lt(pred_rew1, pred_rew2)
                gt_rank = torch.lt(rew1, rew2)
                acc_counter += torch.sum(pred_rank==gt_rank)
                counter += bs1
                if iter_ > 10000:
                    break
            print('Epoch {}, Acc {}'.format(epoch, acc_counter/counter))
            if acc_counter/counter > best_acc:
                best_acc = acc_counter/counter
                # save reward model
                torch.save(reward_net.state_dict(), output_model_path)

        for iter_, data in enumerate(train_loader):
            traj1, rew1, traj2, rew2 = data
            if use_gpu:
                traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
            bs1 = len(traj1)
            bs2 = len(traj2)
            assert bs1 == bs2

            optimizer.zero_grad()
            pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
            pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
            reward_sum = torch.cat([pred_rew1, pred_rew2], dim=1)
            rank_label = (torch.lt(rew1, rew2)).long()

            # calculate the cross entropy loss between reward sum and rank_lable
            loss = F.cross_entropy(reward_sum, rank_label)

            # backpropagate the loss
            loss.backward()
            optimizer.step()

            #print(iter_)
            if iter_ % log_interval == 0:
                print('epoch {}, iter {}, training loss {}'.format(epoch, iter_, loss.item()))
            if iter_ > 5000:
                break
