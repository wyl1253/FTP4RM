import os
import argparse
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
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed

# from torch.utils.tensorboard import SummaryWriter


# os.environ['CUDA_VISIBLE_DEVICE']='1'

title = 'PyBullet UR5e robot'

def train(argv):
    print("============================================================================================")
    
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100  # max timesteps in one episode
    max_episodes = int(1.5e4) #int(1.5e4)
    max_training_timesteps = int(max_episodes*max_ep_len)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 1000  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = print_freq  # save model frequency (in num timesteps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
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
    # args = getopt.getopt(argv, )
    random_seed = int(argv[0])  # set random seed if required (0 = no random seed)
    # random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    ##############Tensorboard##################

    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writepath = 'runs/{}'.format("Catch") + timenow
    if os.path.exists(writepath): shutil.rmtree(writepath)
    # writer = SummaryWriter(log_dir=writepath)

    ################ Pybullet parameters ################
    render = False # render the environment
    repeat = 100 # repeat action 100
    task = 0 # task to learn: 0 move, 1 pick-up, 2 drop
    randObjPos = True # fixed object position to pick up
    simgrip = True # simulated gripper
    lp = 0.0001 # learning parameter for task
    #####################################################

    env_name = title
    print(CP_G + 'Environment name:', env_name, ''+CP_C)
    print("training environment name : " + env_name)

    env = ur5eGymEnv(renders=render, maxSteps=max_ep_len, 
            actionRepeat=repeat, task=task, randObjPos=randObjPos,
            simulatedGripper=simgrip, learning_param=lp)
    
    env.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # state space dimension
    state_dim = 25 #6+7+7+2+3

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
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
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

    replay_buffer = ReplayBuffer(batch_size, state_dim, action_dim)

    # initialize a PPO agent
    ppo_agent = PPO_tricks(batch_size, policy_dist, state_dim, action_dim, max_action, hidden_width, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, max_train_steps, action_std)
    state_norm = Normalization(state_dim)  # Trick 2:state normalization

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, policy_dist, 1, random_seed))


    if use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=gamma)

    if os.path.exists(actor_checkpoint_path):
        ppo_agent.actor_load(actor_checkpoint_path)
    if os.path.exists(critic_checkpoint_path):
        ppo_agent.critic_load(critic_checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # target positions file
    target_f = open(target_f_name, "w+")
    target_f.write('timestep,target_position_x, target_position_y, target_position_z\n')

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

    for i_episode in range(1, max_episodes+1):
        state = env.reset()

        # print(state)
        # print(state[15:18])
        current_ep_reward = 0
        actions_0 = []
        actions_1 = []
        actions_2 = []
        actions_3 = []
        actions_4 = []
        actions_5 = []
        at = []

        if use_state_norm:
            state = state_norm(state)
        if use_reward_scaling:
            reward_scaling.reset()


        for t in range(1, max_ep_len + 1):

            action, a_logprob = ppo_agent.choose_action(state, i_episode, max_episodes)


            if policy_dist == "Beta":
                action = 2 * (action - 0.5) * max_action  # [0,1]->[-max,max]
            else:
                action = action

           
            # state_, reward, done, _ = env.step(action, i_episode, max_episodes, lp)
            joint_angles = np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0.00])

            joint_angles += action

            state_, reward, done, _ = env.step(joint_angles, i_episode, max_episodes, lp)

            if use_state_norm:
                state_ = state_norm(state_)
            if use_reward_norm:
                reward = reward_norm(reward)
            elif use_reward_scaling:
                reward = reward_scaling(reward)

            if done and t != max_ep_len + 1:
                dw = True
            else:
                dw = False

            replay_buffer.store(state, action, a_logprob, reward, state_, dw, done)

            state = state_

            time_step += 1
            current_ep_reward += reward

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == batch_size:
                a_loss, c_loss, entropy = ppo_agent.update(replay_buffer, time_step)
                if a_loss.ndim != 0:  # check if a_loss is not a scalar
                    a_loss = a_loss.mean()  # or .sum(), .item(), etc.
                if c_loss.ndim != 0:  # check if c_loss is not a scalar
                    c_loss = c_loss.mean()  # or .sum(), .item(), etc.
                if entropy.ndim != 0:  # check if entropy is not a scalar
                    entropy = entropy.mean()  # or .sum(), .item(), etc.

                # writer.add_scalar('a_loss', a_loss, global_step=time_step)
                # writer.add_scalar('c_loss', c_loss, global_step=time_step)
                # writer.add_scalar('entropy', entropy, global_step=time_step)

                replay_buffer.count = 0

            # # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = np.round(log_avg_reward, 4)
            
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
            
                log_running_reward = 0
                log_running_episodes = 0

            # # if time_step % max_ep_len == 0:
            target_f.write('{},{},{},{},{}\n'.format(time_step, state[-4], state[-3], state[-2], state[-1]))
            target_f.flush()

            stateaction_f.write('{},{},{}\n'.format(time_step, action, state_))
            stateaction_f.flush()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = np.round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                # print("--------------------------------------------------------------------------------------------")
                # print("saving model at : " + actor_checkpoint_path)
                ppo_agent.actor_save(actor_checkpoint_path)
                # print("model saved")
                # print("saving model at : " + critic_checkpoint_path)
                ppo_agent.critic_save(critic_checkpoint_path)
                # print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                # print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break




        # if time_step % max_ep_len == 0:
        # target_f.write('{},{},{},{},{}\n'.format(time_step, state[9], state[10], state[11], state[12]))
        # target_f.flush()

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
    # i_epoch += 1

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
    train(sys.argv[1:])
