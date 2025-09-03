# '''
# Behavior cloning MLE(Learnt variance) and (MSE)Fixed variance policy.
# '''

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML

from common.sac import ReplayBuffer, SAC

import envs
from utils import system, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy
from math import pi


def try_evaluate(itr: int, policy_type: str):
    assert policy_type in ["Running"]
    # update_time = itr * v['bc']['eval_freq']
    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], True)
    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))
    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], False)
    print(f"real sto return avg: {real_return_sto:.2f}")
    return real_return_det, real_return_sto


def mse_bc(sac_agent, expert_states, expert_actions, epochs = 100):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 1000
    total_loss = 0
    for i in range(epochs):
        for batch_no in range(expert_states.shape[0]//batch_size):
            start_id = batch_no*batch_size
            end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
            se = ((sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]))[0] - torch.FloatTensor(expert_actions[start_id:end_id,:]))**2).sum(1)
            loss = se.mean()
            sac_agent.pi_optimizer.zero_grad()
            total_loss+=se.sum()
            loss.backward()
            sac_agent.pi_optimizer.step()

    total_loss = total_loss/(epochs*expert_states.shape[0])

    return total_loss               

def MLE_bc(sac_agent, expert_states, expert_actions, epochs = 100, reg_coeff=1e-4):
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 128
    total_loss = 0
    for i in range(epochs):
        for batch_no in range(expert_states.shape[0]//batch_size):
            start_id = batch_no*batch_size
            end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
            #print(torch.FloatTensor(expert_states[start_id:end_id,:]).shape) #1000*17
            #print(sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]))[0].shape) #1000*6
            
            #se = ((sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]))[0] - torch.FloatTensor(expert_actions[start_id:end_id,:]))**2).sum(1)
            #print(type(sac_agent.ac.pi))
            action_mu,action_std = sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]), use_std=True) #1000*6,1000
            #action_std=action_std.reshape(action_std.shape[0],1)
            #print('mu',action_mu.shape) #batch *action
            #print('std',action_std.shape) #batch *action
            
            #print(((torch.FloatTensor(expert_actions[start_id:end_id,:])-action_mu)/ torch.exp(action_std)).shape)
            likelihood = -0.5*torch.square((torch.FloatTensor(expert_actions[start_id:end_id,:])-action_mu)) / torch.exp(action_std) - action_std - 0.5 * torch.log(torch.FloatTensor([2 * pi]))
            likelihood = -torch.sum(likelihood,1).reshape(likelihood.shape[0],1)
            regularization_loss=torch.square(action_mu).sum() + torch.square(action_std).sum()
            #print('likelihood loss:',likelihood,'regularization_loss:',regularization_loss)
            loss = likelihood.mean() + reg_coeff*regularization_loss
            sac_agent.pi_optimizer.zero_grad()
            total_loss+=likelihood.sum()
            loss.backward()
            sac_agent.pi_optimizer.step()

    total_loss = total_loss/(epochs*expert_states.shape[0])

    return total_loss               
    

def warm_start(sac_agent, expert_states, expert_actions,v=None, epochs = 100, env_fn=None, reg_coeff=2*1e-5):
    
    assert expert_states.shape[0] == expert_actions.shape[0]
    batch_size = 128
    print('Warm Start Begins')
    for itr in range(v['bc']['epochs']//v['bc']['eval_freq']):
        print('itr:',itr)
        real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], True)
        print(f"real det return avg: {real_return_det:.2f}")
    
        real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['bc']['eval_episodes'], v['env']['T'], False)
        print(f"real sto return avg: {real_return_sto:.2f}")
        for i in range(epochs):
            for batch_no in range(expert_states.shape[0]//batch_size):
                start_id = batch_no*batch_size
                end_id = min((batch_no+1)*batch_size,expert_states.shape[0])
                action_mu,action_std = sac_agent.ac.pi(torch.FloatTensor(expert_states[start_id:end_id,:]), use_std=True) 
                likelihood = -0.5*torch.square(torch.FloatTensor(expert_actions[start_id:end_id,:])-action_mu) / torch.exp(action_std) - action_std - 0.5 * torch.log(torch.FloatTensor([2 * pi]))
                likelihood = -torch.sum(likelihood,1).reshape(likelihood.shape[0],1)
                regularization_loss=torch.square(action_mu).sum() + torch.square(action_std).sum()
                
                loss = likelihood.mean() + reg_coeff*regularization_loss
                sac_agent.pi_optimizer.zero_grad()
                loss.backward()
                sac_agent.pi_optimizer.step() 
            print('likelihood loss:',likelihood.mean(),'regularization_loss:',regularization_loss)


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['bc']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['bc']
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    state_indices = list(range(state_size))
    action_indices = list(range(action_size))
    
    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_state_trajs=np.load('./halfcheetah_Pdata/11500/halfcheetah_medium_states.npy')[:num_expert_trajs, :, :]
    expert_action_trajs=np.load('./halfcheetah_Pdata/11500/halfcheetah_medium_actions.npy')[:num_expert_trajs, :, :]
    expert_state_samples = expert_state_trajs.copy().reshape(-1, len(state_indices))
    expert_action_samples = expert_action_trajs.copy().reshape(-1, len(action_indices))
    # concatenate state and action
    
    replay_buffer = ReplayBuffer(
                    state_size, 
                    action_size,
                    device=device,
                    size=v['sac']['buffer_size'])
    sac_agent = SAC(env_fn, replay_buffer,
        steps_per_epoch=v['env']['T'],
        update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
        max_ep_len=v['env']['T'],
        seed=seed,
        start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
        reward_state_indices=state_indices,
        device=device,
        **v['sac']
    )


    for itr in range(v['bc']['epochs']//v['bc']['eval_freq']):
        loss = MLE_bc(sac_agent, expert_state_samples, expert_action_samples, epochs = v['bc']['eval_freq'])
        logger.record_tabular("BC loss", loss.item())

        real_return_det, real_return_sto = try_evaluate(itr, "Running")
        logger.record_tabular("Iteration", itr)
        logger.dump_tabular()
