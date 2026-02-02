import os
import argparse
from re import L
import time
from datetime import datetime
import numpy as np
from itertools import count
from collections import namedtuple, deque
import pickle
import torch
import gym
import random
import sys, getopt


from ppo_tricks import PPO_tricks
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from gym_env import ur5eGymEnv

from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from statistics import mean
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


# os.environ['CUDA_VISIBLE_DEVICE']='1'

title = 'PyBullet UR5e robot'

def test(argv):
    print("============================================================================================")
    
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100  # max timesteps in one episode
    max_episodes = int(100)
    max_epochs = 19
    max_training_timesteps = max_ep_len*max_episodes # break training loop if timeteps > max_training_timesteps
    print_freq = max_episodes * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    action_std = 1  # starting std for action distribution (Multivariate Normal)
    min_action_std = 0.6  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_rate = 0  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    action_std_decay_freq = int(1e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    # Color Palette
    CP_R = '\033[31m'
    CP_G = '\033[32m'
    CP_B = '\033[34m'
    CP_Y = '\033[33m'
    CP_C = '\033[0m'

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    max_action = 3.14
    hidden_width = 256
    lamda = 0.95
    batch_size = 4096
    max_train_steps = max_training_timesteps

    K_epochs = 20  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.0003  # learning rate for critic network
    # random_seed = int(argv[0])  # set random seed if required (0 = no random seed)

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################
    
    ################ Pybullet parameters ################
    render = False # render the environment
    repeat = 100 # repeat action
    task = 0 # task to learn: 0 move, 1 pick-up, 2 drop
    randObjPos = True # fixed object position to pick up
    simgrip = True # simulated gripper
    lp = 0.005 # learning parameter for task
    #####################################################

    env_name = title
    print(CP_G + 'Environment name:', env_name, ''+CP_C)
    print("testing environment name : " + env_name)
    
    env = ur5eGymEnv(renders=render, maxSteps=max_ep_len, 
            actionRepeat=repeat, task=task, randObjPos=randObjPos,
            simulatedGripper=simgrip, learning_param=lp)
    
    env.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # state space dimension
    state_dim = 25

    # action space dimension
    if has_continuous_action_space:
        action_dim = 6
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_tricks_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_Tricks' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    #### log files for target_positions
    target_dir = "Target_positions"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_dir = target_dir + '/' + env_name + '/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    #### get number of log files in log directory
    target_run_num = 0
    current_num_files_target = next(os.walk(target_dir))[2]
    target_run_num = len(current_num_files_target)

    #### create new log file for each run
    target_f_name = target_dir + '/Target_' + env_name + "_log_" + str(target_run_num) + ".csv"
    #####################################################

    #### log files for statesactions
    stateaction_dir = "States_Actions"
    if not os.path.exists(stateaction_dir):
        os.makedirs(stateaction_dir)

    stateaction_dir = stateaction_dir + '/' + env_name + '/'
    if not os.path.exists(stateaction_dir):
        os.makedirs(stateaction_dir)

    #### get number of log files in log directory
    stateaction_run_num = 0
    current_num_files_stateaction = next(os.walk(stateaction_dir))[2]
    stateaction_run_num = len(current_num_files_stateaction)

    #### create new log file for each run
    stateaction_f_name = stateaction_dir + '/stateaction_' + env_name + "_log_" + str(stateaction_run_num) + ".csv"
    #####################################################


    # ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_Tricks_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    actor_checkpoint_path = directory + "PPO_Tricks_actor_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save actor_checkpoint path : " + actor_checkpoint_path)

    critic_checkpoint_path = directory + "PPO_Tricks_critic_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save critic_checkpoint path : " + critic_checkpoint_path)
    # #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    # print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    # print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    ################# training procedure ################
    use_reward_norm = False
    use_reward_scaling = False
    use_state_norm = False
    policy_dist = "Gaussian" # Gaussian or Beta

    # replay_buffer = ReplayBuffer(batch_size, state_dim, action_dim)

    # initialize a PPO agent
    # ppo_agent = PPO_tricks(batch_size, policy_dist, state_dim, action_dim, max_action, hidden_width, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, max_train_steps, action_std)
    state_norm = Normalization(state_dim)  # Trick 2:state normalization

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, policy_dist, 1, random_seed))


    if use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=gamma)

    # ppo_agent.actor_load(actor_checkpoint_path)
    # ppo_agent.critic_load(critic_checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # target positions file
    target_f = open(target_f_name, "w+")
    # target_f.write('timestep,target_position_x, target_position_y, target_position_z\n')
    target_f.write('error,success_rate\n')


    # stateaction positions file
    stateaction_f = open(stateaction_f_name, "w+")
    stateaction_f.write('timestep, actions, states\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    i_epoch = 0
    success_count = 0
    success_rate = 0


    model = A2C("MlpPolicy", env, verbose=1, seed = random_seed)
    model_name = "a2c" + str(argv[0])
    model = A2C.load(model_name)

    for i_epoch in range(0, max_epochs+1):

        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            current_ep_reward = 0

            if use_state_norm:
                state = state_norm(state)
            if use_reward_scaling:
                reward_scaling.reset()

            for t in range(1, max_ep_len + 1):

                # select action with policy
                action, a_logprob = model.predict(state)

                if policy_dist == "Beta":
                    action = 2 * (action - 0.5) * max_action  # [0,1]->[-max,max]
                else:
                    action = action


                state_, reward, done, _ = env.step(action, lp)
                # print(state_)
                if use_state_norm:
                    state_ = state_norm(state_)
                if use_reward_norm:
                    reward = reward_norm(reward)
                elif use_reward_scaling:
                    reward = reward_scaling(reward)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and t != max_ep_len + 1:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）

                # print(reward)


                # if reward < 0.5:
                #     replay_buffer.store(state, action, a_logprob, reward, state_, dw, done)


                state = state_

                time_step += 1
                current_ep_reward += reward

                # When the number of transitions in buffer reaches batch_size,then update

                # # log in logging file
                # if time_step % log_freq == 0:
                #     # log average reward till last episode
                #     log_avg_reward = log_running_reward / log_running_episodes
                #     log_avg_reward = np.round(log_avg_reward, 4)
                #
                #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                #     log_f.flush()
                #
                #     log_running_reward = 0
                #     log_running_episodes = 0

                # if time_step % max_ep_len == 0:
                #     target_f.write('{},{},{},{},{}\n'.format(time_step, state[9], state[10], state[11], state[12]))
                #     target_f.flush()

                stateaction_f.write('{},{},{}\n'.format(time_step, action, state_))
                stateaction_f.flush()

                # printing average reward
                # if time_step % print_freq == 0:
                #     # print average reward till last episode
                #     print_avg_reward = print_running_reward / print_running_episodes
                #     print_avg_reward = np.round(print_avg_reward, 2)

                #     print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                #                                                                             print_avg_reward))

                #     print_running_reward = 0
                #     print_running_episodes = 0

                # break; if the episode is over
                if done:
                    log_f.write('{},{},{}\n'.format(i_episode, t, lp))
                    log_f.flush()
                    success_count += 1
                    break

            # target_f.write('{},{},{},{},{}\n'.format(i_epoch, state[9], state[10], state[11], state[12]))
            # target_f.flush()

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        success_rate = success_count/max_episodes
        target_f.write('{},{}\n'.format(lp, success_rate))
        target_f.flush()
        success_count = 0
        lp += 0.005


        i_epoch += 1


    log_f.close()
    target_f.close()
    stateaction_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    test(sys.argv[1:])
