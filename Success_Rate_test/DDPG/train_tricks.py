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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# from torch.utils.tensorboard import SummaryWriter

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

def train(argv):
    print("============================================================================================")
    
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100  # max timesteps in one episode
    max_episodes = int(1.5e4)
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

    # timenow = str(datetime.now())[0:-10]
    # timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    # writepath = 'runs/{}'.format("Catch") + timenow
    # if os.path.exists(writepath): shutil.rmtree(writepath)
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


    class SaveOnBestTrainingRewardCallback(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq:
        :param log_dir: Path to the folder where the model will be saved.
          It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: Verbosity level.
        """
        def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, 'best_model')
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

              # Retrieve training reward
              x, y = ts2xy(load_results(self.log_dir), 'timesteps')
              # print(x,y)


              if len(x) > 0:
                  log_f.write('{},{}\n'.format(x[-1],y[-1]))
                  log_f.flush()
                  # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  # print(y[-100:])
                  if self.verbose > 0:
                      pass
                    # print(f"Num timesteps: {self.num_timesteps}")
                    # print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                  # New best model, you could save the agent here
                  if mean_reward > self.best_mean_reward:
                      self.best_mean_reward = mean_reward
                      # Example for saving best model
                      if self.verbose > 0:
                          pass
                        # print(f"Saving new best model to {self.save_path}")
                      # self.model.save(self.save_path)

            return True


    class RewardLoggerCallback(BaseCallback):
        def __init__(self, log_file, accumulate_steps=100, verbose=0):
            super(RewardLoggerCallback, self).__init__(verbose)
            self.log_file = log_file
            self.accumulate_steps = accumulate_steps
            self.accumulated_reward = 0
            self.step_count = 0
            # Ensure the file is ready for writing
            with open(self.log_file, "w+") as log_f:
                log_f.write('timestep,reward\n')

        def _on_step(self) -> bool:
            self.step_count += 1
            self.accumulated_reward += self.locals['rewards'][0]  # Accumulate rewards

            # Write to the log file every `accumulate_steps`
            if self.step_count % self.accumulate_steps == 0:
                timestep = self.num_timesteps
                with open(self.log_file, "a+") as log_f:
                    log_f.write(f"{timestep},{self.accumulated_reward}\n")
                # Reset accumulated reward
                self.accumulated_reward = 0
            
            return True



    start_time = datetime.now().replace(microsecond=0)

   ######## DDPG ########
    log_f = open(log_f_name, "w+")
    log_f.write('timestep,reward\n')
    env = DummyVecEnv([lambda: env])
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, seed = random_seed)
    custom_reward_logger = RewardLoggerCallback(log_file=log_f_name, accumulate_steps=100)
    model.learn(total_timesteps=1500000, callback=custom_reward_logger)

    model_name = "ddpg" + str(argv[0])
    model.save(model_name)
    del model # remove to demonstrate saving and loading
    model = DDPG.load(model_name)




    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train(sys.argv[1:])
