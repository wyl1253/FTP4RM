from xml.etree import ElementInclude
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal, MultivariateNormal
from scipy.stats import poisson, chi2, binom, beta
import numpy as np
import random


################################## set device ##################################
# print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
    # print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    pass
    # print("Device set to : cpu")
# print("============================================================================================")


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, state_dim, hidden_width, action_dim):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.alpha_layer = nn.Linear(hidden_width, action_dim)
        self.beta_layer = nn.Linear(hidden_width, action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][True]  # Trick10: use tanh

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1
        beta = F.softplus(self.beta_layer(s)) + 1
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)

        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, max_action, state_dim, hidden_width, action_dim, action_std_init):
        super(Actor_Gaussian, self).__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.fc1 = nn.Linear(state_dim,hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim).to(device))  # We use 'nn.Paremeter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][True]  # Trick10: use tanh

        # use_orthogonal:
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s)).to(device)  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        # Our method variance setting
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std).to(device)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        # the original PPO variance setting
        # std = self.action_var
        # std = std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        # cov_mat = torch.diag_embed(std).to(device)
        # dist = MultivariateNormal(mean, cov_mat)
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][True]  # Trick10: use tanh


        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_tricks():
    def __init__(self, batch_size, policy_dist, state_dim, action_dim, max_action, hidden_width, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, max_train_steps, action_std_init):

        self.policy_dist = policy_dist # Beta or Gaussian
        self.max_action = max_action
        self.batch_size = batch_size
        self.mini_batch_size = int(batch_size*0.25)
        self.max_train_steps = max_train_steps
        self.lr_a = lr_actor  # Learning rate of actor
        self.lr_c = lr_critic  # Learning rate of critic
        self.gamma = gamma  # Discount factor
        self.lamda = lamda  # GAE parameter
        self.epsilon = eps_clip  # PPO clip parameter
        self.K_epochs = K_epochs  # PPO parameter
        self.entropy_coef = 0.01  # Entropy coefficient Trick 5: policy entropy
        self.set_adam_eps = False
        self.use_grad_clip = False
        self.use_lr_decay = True
        self.use_adv_norm = False
        self.action_std = action_std_init

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(state_dim, hidden_width, action_dim).to(device)
        else:
            self.actor = Actor_Gaussian(max_action, state_dim, hidden_width, action_dim, action_std_init).to(device)
        self.critic = Critic(state_dim, hidden_width).to(device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # for storing weights
        if self.policy_dist == "Beta":
            self.actor_old = Actor_Beta(state_dim, hidden_width, action_dim).to(device)
        else:
            self.actor_old = Actor_Gaussian(max_action, state_dim, hidden_width, action_dim, action_std_init).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic_old = Critic(state_dim, hidden_width).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict())


    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(device), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)
        self.actor_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            # print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            pass
            # print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)


    def choose_action(self, s, i_episode, max_episodes):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(device), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                # a = dist.sample()  # Sample the action according to the probability distribution

                # Improvement 1

                # Beta 
                alpha = 1 + 5 * (i_episode / max_episodes)  # alpha increases faster
                beta_param = 1 + 20 * (i_episode / max_episodes)  # beta increases, but at a slower rate
                
                n = 10
                i = np.clip(n*beta.rvs(alpha, beta_param), 1, alpha)
                aa = dist.sample((int(i),))
                aa_a = aa.cpu().data.numpy()
                a = aa_a.mean(axis=0)
                a = torch.tensor(a).to(device)

                # n = 5
                # alpha = n + 1 - n * (i_episode / max_episodes)  
                # beta_param = 1 + n * (i_episode / max_episodes)
                
                # i = np.clip(n*beta.rvs(alpha, beta_param), 1, alpha)
                # aa = dist.sample((int(i),))
                # aa_a = aa.cpu().data.numpy()
                # a = aa_a.mean(axis=0)
                # a = torch.tensor(a).to(device)

                # # Exponential
                # lambda_rate = 1 / (1 + 5 * (i_episode / max_episodes))  # Define the rate parameter for the Exponential distribution
                # sample = expon.rvs(scale=1/lambda_rate)  # 'scale' is the inverse of 'lambda_rate' (i.e., mean of the distribution)
                # sample = np.clip(sample, 1, np.inf)  # Ensure sample is at least 1 or higher
                # aa = dist.sample((int(sample),))
                # aa_a = aa.cpu().data.numpy()
                # a = aa_a.mean(axis=0)
                # a = torch.tensor(a).to(device)

                # # Linear
                # beta = 1 + 5*(i_episode/max_episodes)
                # i = beta
                # aa = dist.sample((int(i),))
                # aa_a = aa.cpu().data.numpy()
                # a = aa_a.mean(axis=0)
                # a = torch.tensor(a).to(device)


                # # Poisson
                # beta = 1 + 5*(i_episode/max_episodes)
                # i = np.clip(poisson.rvs(mu=beta), 1, beta)
                # aa = dist.sample((int(i),))
                # aa_a = aa.cpu().data.numpy()
                # a = aa_a.mean(axis=0)
                # a = torch.tensor(a).to(device)

                

##################################3

                a = torch.clamp(a, -self.max_action, self.max_action).to(device)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action

        return a.cpu().data.numpy().flatten(), a_logprob.cpu().data.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        # np.nan_to_num()
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0

        # Improvement 2
        zero = torch.zeros_like(a_logprob)
        tanhalog = torch.tanh(0.1*a_logprob)
        expalog = torch.exp(a_logprob)

        pisa = torch.where(a_logprob > zero, tanhalog, expalog)
        new_gamma = pisa.mean(1, keepdim=True)

        # pisa = torch.exp(a_logprob.mean(1, keepdim=True)).to(device)
        self.gamma = torch.clamp(new_gamma, 0.6, 0.99).to(device)

        # the original PPO and PPO_AEP
        # self.gamma = torch.clamp(pisa, 0.99, 0.99).to(device)
        # self.gamma = torch.clamp(torch.exp(a_logprob.mean(1, keepdim=True)).to(device), 0.55, 0.99).to(device)
        # self.gamma = torch.clamp(torch.exp(-5*torch.pow(a_logprob.mean(1, keepdim=True), 2).to(device)).to(device), 0.55, 1).to(device)

        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs

            for delta, d, gamma in zip(reversed(deltas.cpu().data.flatten().numpy()), reversed(done.cpu().data.flatten().numpy()), reversed(self.gamma.cpu().data.flatten().numpy())):
                gae = delta + gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).to(device).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                # print(dist_now)
                a_logprob_now = dist_now.log_prob(a[index])

                # # Improvement 3
                # beta4 = 0.5
                # ee4 = 0.2
                # delta4 = torch.sign(adv[index])
                # pp = a_logprob_now.sum(1, keepdim=True)/a_logprob[index].sum(1, keepdim=True)
                # ppb = pp/(1-beta4+beta4*pp)
                # denom = delta4*ee4*(beta4*delta4*ee4-2*(1-beta4*(1+delta4*ee4)))
                # numer = adv[index]*(ppb-1)*(beta4*(ppb-1)-2*(1-beta4)/(1-beta4+beta4*pp))
                # # adv[index] = torch.tensor(adv[index] - numer/denom, dtype=torch.float).to(device).view(-1, 1)
                # advv = adv[index] - numer/denom
                # adv[index] = advv.clone().detach().view(-1, 1)


                # Improvement 5
                # lamda5 = random.random()*0.5
                # lamda5 = 0.8
                # deltav = torch.sign(vs_[index]-vs[index]) * lamda5
                # im5 = 1+deltav

                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)
                
                # ratios = ratios*im5
                
                
                surr1 = ratios * adv[index]   # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                ppc = torch.sqrt(torch.exp(a_logprob[index].mean(1, keepdim=True)))
                v_s = self.critic(s[index]) 
                
                # print(v_s)
                # print(torch.exp(a_logprob[index].mean(1, keepdim=True)))
                critic_loss = F.mse_loss(v_target[index], v_s)
                # print(critic_loss)

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

            # Copy new weights into old policy
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critic_old.load_state_dict(self.critic.state_dict())

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        return actor_loss, critic_loss, dist_entropy


    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def actor_save(self, checkpoint_path):
        torch.save(self.actor_old.state_dict(), checkpoint_path)

    def actor_load(self, checkpoint_path):
        self.actor_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def critic_save(self, checkpoint_path):
        torch.save(self.actor_old.state_dict(), checkpoint_path)

    def critic_load(self, checkpoint_path):
        self.actor_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
