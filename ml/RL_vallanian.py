import sys, os, time
from ruamel.yaml import YAML
from utils import system

import gym
import numpy as np 
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import envs
from common.sac import ReplayBuffer, SAC
from ml.models.reward import MLPReward
from utils.plots.train_plot import plot_sac_curve

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--config', metavar='N',default=None,type=str,
                    help='an integer for the accumulator')
parser.add_argument('--Hidden_size', metavar='N',default=64,type=int,
                    help='an integer for the accumulator')
parser.add_argument('--policy_model_path', metavar='N',default=None, type=str,
                    help='an integer for the accumulator')
parser.add_argument('--output', metavar='N',default="./save_policy/stage1", type=str,
                    help='an integer for the accumulator')
args = parser.parse_args()

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(args.config))

    # common parameters
    env_name, env_T = v['env']['env_name'], v['env']['T']
    state_indices = v['env']['state_indices']
    seed = int(time.time())
    if args.policy_model_path:
        model=torch.load(args.policy_model_path)
        print(model["pi.net.2.weight"].shape[0])
        args.Hidden_size = model["pi.net.2.weight"].shape[0]
    v['sac']['hidden_sizes'][1] = args.Hidden_size
    if args.Hidden_size <64:
        v['sac']['random_explore_episodes']=0
    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    train_env = gym.make(env_name)
    test_env = gym.make(env_name) # original reward

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    print(f"Transfer: training Expert on {env_name}")

    replay_buffer = ReplayBuffer(
        gym_env.observation_space.shape[0], 
        gym_env.action_space.shape[0],
        device=device,
        size=v['sac']['buffer_size'])
    
    sac_agent = SAC(env_fn, replay_buffer,
        steps_per_epoch=env_T,
        update_after=30000, 
        max_ep_len=env_T,
        seed=seed,
        start_steps=0,
        device=device,
        **v['sac']
        )
    sac_agent.reinitialize=True
    assert sac_agent.reinitialize == True
    if args.policy_model_path:
        sac_agent.ac.load_state_dict(torch.load(args.policy_model_path))
        print(f'have loaded the model {args.policy_model_path}')
    sac_agent.env = train_env
    sac_agent.test_env = test_env
    sac_agent.test_fn = sac_agent.test_agent_ori_env
    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_agent.learn_mujoco(print_out=True, save_path=f'{args.output}/{env_name}.pt')
    plot_sac_curve(ax, sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)
    os.makedirs("expert_data/optimal_policy/", exist_ok=True)
    plt.savefig(os.path.join(f"expert_data/optimal_policy/gd_{env_name}-{seed}.pdf"))
    log_txt = open(f"expert_data/optimal_policy/gd_{env_name}_{seed}.txt", 'w')
    log_txt.write(repr(sac_test_rets)+'\n')