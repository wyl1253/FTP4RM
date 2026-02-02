import torch
import numpy as np


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




class ReplayBuffer:
    def __init__(self, batch_size, state_dim, action_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, action_dim))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(device)
        a = torch.tensor(self.a, dtype=torch.float).to(device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        r = torch.tensor(self.r, dtype=torch.float).to(device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        done = torch.tensor(self.done, dtype=torch.float).to(device)

        return s, a, a_logprob, r, s_, dw, done
