import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from SparkENV import SparkENV
#####################  hyper parameters  ####################

MAX_EPISODES = 50
MAX_EP_STEPS = 100
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.5  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 100
num_iteration = 10
noise_clip = 0.5
policy_delay = 2
policy_noise = 0.2
directory="../"
# directory="../model_td3_er/wc/"
# directory="../model_ts_no_der_done/"
# directory="../model_ts_increase/"
# c=520




###############################  TD3   ####################################

class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, a_dim)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(256)
        self.bn3=nn.BatchNorm1d(a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class CNet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 128)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(a_dim, 128)
        self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.fc = nn.Linear(128, 128)
        self.fc.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.bn1 = nn.BatchNorm1d(128)


    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        net = self.fc(net)
        net = self.bn1(net)
        actions_value = self.out(net)
        return actions_value


class TD3(object):
    def __init__(self, a_dim, s_dim,scaler):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.memory2 = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.pointer2= 0
        self.max_action=1
        self.scaler = scaler

        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic1_eval = CNet(s_dim, a_dim)
        self.Critic2_eval = CNet(s_dim, a_dim)
        self.Critic1_target = CNet(s_dim, a_dim)
        self.Critic2_target = CNet(s_dim, a_dim)

        self.Critic1_target.load_state_dict(self.Critic1_eval.state_dict())
        self.Critic2_target.load_state_dict(self.Critic2_eval.state_dict())
        self.Actor_target.load_state_dict(self.Actor_eval.state_dict())

        self.ctrain1 = torch.optim.Adam(self.Critic1_eval.parameters(), lr=LR_C)
        self.ctrain2 = torch.optim.Adam(self.Critic2_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        # self.loss_td = nn.MSELoss()


    def  choose_action(self, s):   # s []
        s = normalization(s, 1,self.scaler)     # s [[]] for normalize
        s = torch.FloatTensor(s)

        self.Actor_eval.eval()   # when BN or Dropout in testing ################
        self.Critic1_eval.eval()   # when BN or Dropout in testing ################
        self.Critic2_eval.eval()   # when BN or Dropout in testing ################
        # s = torch.unsqueeze(torch.FloatTensor(s), 0)
        act = self.Actor_eval(s)[0].detach()  # ae（s）
        print('original action',act)
        starttime = datetime.datetime.now()
        # twin-Q optimizer
        c=0
        while(c<50000):
            q1 = self.Critic1_eval(s, act).detach()
            q2 = self.Critic2_eval(s, act).detach()
            q = torch.min(q1, q2)
            if q.item()>=0.1:
                break
            else:
                act = np.clip(np.random.normal(act, 0.1), 0, 1)
                act = torch.FloatTensor(act)
                c += 1
        endtime = datetime.datetime.now()
        duringtime = endtime - starttime
        print('q=',q.item())
        print(f'change action {c} time,consume {duringtime}')
        self.Actor_eval.train()   
        self.Critic1_eval.train()  
        self.Critic2_eval.train()  
        return act

    def learn(self):
        self.Actor_eval.train()    # #################
        for i in range(num_iteration):
            # =================RDPER=======================
            # memory 1
            indices1 = np.random.choice(self.pointer, size=)
            bt1 = self.memory[indices1, :]
            bs1 = torch.FloatTensor(bt1[:, :self.s_dim])
            ba1 = torch.FloatTensor(bt1[:, self.s_dim: self.s_dim + self.a_dim])
            br1 = torch.FloatTensor(bt1[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim + 1])
            bs_1 = torch.FloatTensor(bt1[:, -self.s_dim - 1:-1])
            bd1 = torch.FloatTensor(bt1[:, -1:])
            # memory 2
            indices2 = np.random.choice(self.pointer2, size=50)
            bt2 = self.memory2[indices2, :]
            bs2 = torch.FloatTensor(bt2[:, :self.s_dim])
            ba2 = torch.FloatTensor(bt2[:, self.s_dim: self.s_dim + self.a_dim])
            br2 = torch.FloatTensor(bt2[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim + 1])
            bs_2 = torch.FloatTensor(bt2[:, -self.s_dim - 1:-1])
            bd2 = torch.FloatTensor(bt2[:, -1:])
            
            bs=torch.cat((bs1,bs2),0)
            ba=torch.cat((ba1,ba2),0)
            br=torch.cat((br1,br2),0)
            bs_=torch.cat((bs_1,bs_2),0)
            bd=torch.cat((bd1,bd2),0)

            # =============normal ER=======================

            # indices = np.random.choice(self.pointer, size=64)
            # bt = self.memory[indices, :]
            # bs = torch.FloatTensor(bt[:, :self.s_dim])
            # ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
            # br = torch.FloatTensor(bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim + 1])
            # bs_ = torch.FloatTensor(bt[:, -self.s_dim - 1:-1])
            # bd = torch.FloatTensor(bt[:, -1:])

            bs = normalization(bs, 0,self.scaler)
            bs_ = normalization(bs_, 0,self.scaler)

            a = self.Actor_eval(bs)
            # todo: a' add noise
            noise = torch.ones_like(ba).data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)

            a_ = self.Actor_target(bs_) + noise
            a_ = a_.clamp(0,1)
            q1 = self.Critic1_target(bs_, a_)
            q2 = self.Critic2_target(bs_, a_)
            q_ = torch.min(q1, q2)
            q_target = br + (1-bd)*(GAMMA * q_).detach()  # q_target = 负的

            q_v1 = self.Critic1_eval(bs, ba)
            td_error1 = F.mse_loss(q_target, q_v1)
            self.ctrain1.zero_grad()
            td_error1.backward()
            self.ctrain1.step()

            q_v2 = self.Critic2_eval(bs, ba)
            td_error2 = F.mse_loss(q_target, q_v2)
            self.ctrain2.zero_grad()
            td_error2.backward()
            self.ctrain2.step()

            if i % policy_delay == 0:
                q = self.Critic1_eval(bs, a)  # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
                loss_a = -torch.mean(q)
                self.atrain.zero_grad()
                loss_a.backward()
                self.atrain.step()
                for (target_param,param) in zip(self.Actor_target.parameters(),self.Actor_eval.parameters()):
                    target_param.data.copy_(
                        target_param.data*(1-TAU)+param.data*TAU
                    )
                for (target_param,param) in zip(self.Critic1_target.parameters(),self.Critic1_eval.parameters()):
                    target_param.data.copy_(
                        target_param.data*(1-TAU)+param.data*TAU
                    )
                for (target_param, param) in zip(self.Critic2_target.parameters(), self.Critic2_eval.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1 - TAU) + param.data * TAU
                    )
               

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        if r>=1.6:
            index2 = self.pointer2 % MEMORY_CAPACITY
            self.memory2[index2, :] = transition
            self.pointer2 += 1
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def load_model(self, episode):
        model_name_c1 = "Critic1_td3_" + str(episode) + ".pt"
        model_name_c2 = "Critic2_td3_" + str(episode) + ".pt"
        model_name_a = "Actor_td3_" + str(episode) + ".pt"
        self.Critic1_target = torch.load(directory + model_name_c1)
        self.Critic1_eval = torch.load(directory + model_name_c1)
        self.Critic2_target = torch.load(directory + model_name_c2)
        self.Critic2_eval = torch.load(directory + model_name_c2)
        self.Actor_target = torch.load(directory + model_name_a)
        self.Actor_eval = torch.load(directory + model_name_a)

    def save_model(self, episode):
        model_name_c1 = "Critic1_td3_" + str(episode) + ".pt"
        model_name_c2 = "Critic2_td3_" + str(episode) + ".pt"
        model_name_a = "Actor_td3_" + str(episode) + ".pt"
        torch.save(self.Critic1_eval, directory + model_name_c1)
        torch.save(self.Critic2_eval, directory + model_name_c2)
        torch.save(self.Actor_eval, directory + model_name_a)
       

def normalization(x,typeid,scaler):
    if typeid==1:   # choose action (s)
        x=np.array([x])
        x=scaler.transform(x)
    else:   # batch normal(s)
        x=np.array(x)
        x = scaler.transform(x)
        x=torch.FloatTensor(x)
    return x


def get_scaler():
    x=np.loadtxt('memory/pool_newstate.txt', delimiter=' ')
    x=x[:,:8]  # all s
    standardscaler = StandardScaler()
    scaler=standardscaler.fit(x)    # scaler-> mean,var
    return scaler


def tune(x):
    scaler=get_scaler()
    env = SparkENV()
    s_dim = 8
    a_dim = 32
    count = 0
    td3 = TD3(a_dim, s_dim,scaler)
    td3.load_model(x)
    print(f'============load model ===============')
    td3.memory = np.loadtxt('memory/pool_good.txt', delimiter=' ')
    td3.memory2 = np.loadtxt('memory/pool_bad.txt', delimiter=' ')
    td3.pointer = 200
    td3.pointer2 = 200
    # print('============load memory===============')
    # y = []
    alltime = []
    var = 0.1 # control exploration
    for i in range(1):
        s = env.reset()
        for j in range(10):
            print('s=',s)
            s1=datetime.datetime.now()
            a = td3.choose_action(s)
            s2=datetime.datetime.now()
            print('overhead:',s2-s1)
            s_, r, done, dur = env.step(a,i,j)
            td3.store_transition(s, a, r, s_, done)
            alltime.append(dur)
            s = s_
            print('-------------------------------------------')
            print(alltime)


def offline_train(x):
    scaler = get_scaler()
    # env = SparkENV()
    s_dim = 8
    a_dim = 32
    td3 = TD3(a_dim, s_dim,scaler)
    td3.load_model(x)
    print(f'============load model {x}iters===============')
    td3.memory = np.loadtxt('memory/pool_good.txt', delimiter=' ')
    td3.memory2 = np.loadtxt('memory/pool_bad.txt', delimiter=' ')
    td3.pointer = 1151       
    print('============load memory===============')
    for i in range(4001): #50w
        td3.learn()
    td3.save_model(4000)
      

if __name__=='__main__':
    tune(40000)
   
