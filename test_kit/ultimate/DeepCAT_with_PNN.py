import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from torch.nn import BatchNorm1d
import datetime
from SparkENV import SparkENV
import os
from similar import similar
from scipy.spatial.distance import cdist

MAX_EPISODES = 50
MAX_EP_STEPS = 10
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 1480
BATCH_SIZE = 32
current_directory = os.getcwd()

def get_similar_workload():
    Alog_feature=[]
    target_worklaod_feature=[]
    exist_workload_feature=[]

    file=['wc','ts','pr','km']
    with open(current_directory+'logs/'+'agg.txt') as f:
        seq = f.read()
        for i in range(0,16):
            character_i = str(seq.split('\n')[i])
            Alog_feature.append(float(character_i.split(':')[1]))
    target_worklaod_feature.append(Alog_feature)

    for filename in file:
        with open(current_directory+'logs/'+filename+'.txt') as f:
            base_feature = []
            seq = f.read()
            for i in range(0, 16):
                character_i = str(seq.split('\n')[i])
                base_feature.append(float(character_i.split(':')[1]))
            exist_workload_feature.append(base_feature)

    # print('target_worklaod_feature',target_worklaod_feature)
    # print('exist_workload_feature',exist_workload_feature)

    standardE =cdist(exist_workload_feature, target_worklaod_feature, metric='seuclidean')    #seuclidean
    arr=np.array(standardE)
    min_index =np.argmin(arr)
    wk=file[min_index]
    base_dir=(str(wk).split('.')[0])
    return base_dir

class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, a_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.out.weight.data.normal_(0, 0.1)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2.weight.data.normal_(0, 0.1)



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


class PNN(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(PNN, self).__init__()
        self.layer1 = nn.Linear(s_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, a_dim)
        self.preNET=ANet(s_dim, a_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer1.weight.data.normal_(0, 0.1)  # initialization
        self.layer2.weight.data.normal_(0, 0.1)
        self.layer3.weight.data.normal_(0, 0.1)


    def forward(self, x):
        a1 = self.preNET.fc1(x)
        x = self.layer1(x)
        x = F.relu(x+a1)
        x = self.bn1(x)
        x = self.layer2(x)
        a1=F.relu(a1)
        a1=self.preNET.bn1(a1)
        a2=self.preNET.fc2(a1)    #preNET a1
        x = F.relu(x+a2)
        x = self.bn2(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x


    def freeze_preNET(self):
        for param in self.preNET.parameters():
            param.requires_grad = False



#for new task
class DeepCAT_PNN(object):
    def __init__(self, a_dim, s_dim, scaler):
        super(DeepCAT_PNN, self).__init__()
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory = np.zeros((10, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.policy= PNN(s_dim, a_dim)  
        self.critic1=CNet(s_dim, a_dim) 
        self.critic2=CNet(s_dim, a_dim)
        self.train = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.loss_td = nn.MSELoss()
        self.scaler = scaler

    def choose_action(self, s):  # s []
        s = normalization(s, 1, self.scaler)  # s [[]] for normalize
        s = torch.FloatTensor(s)
        self.policy.eval()  # when BN or Dropout in testing ################
        self.critic1.eval()  # when BN or Dropout in testing ################
        self.critic2.eval()  # when BN or Dropout in testing ################
        act = self.policy(s)[0].detach()  # ae（s）
        self.policy.train() 
        self.critic1.train()  
        self.critic2.train()
        return act

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # def save_model(self, episode):
    #     model_name = "Actor_cdbtune_PNN" + str(episode) + ".pt"
    #     torch.save(self.policy, directory + model_name)

    def load_model(self, dir):
        model_name_a = dir+"Actor_td3.pt"
        model_name_c1 = dir+"Critic1_td3.pt"
        model_name_c2 = dir+"Critic2_td3.pt"
        self.policy.preNET = torch.load(model_name_a)
        self.critic1=torch.load(model_name_c1)
        self.critic2=torch.load(model_name_c2)

    def finetune(self):
        bt = self.memory[:self.pointer,:]
        br = torch.FloatTensor(bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim + 1])
        print(br)
        loss = -torch.mean(br)  
        loss.requires_grad_()
        self.train.zero_grad()
        loss.backward()
        self.train.step()


def normalization(x,typeid,scaler):
    # x:FloatTensor to array normalize then back to FloatTensor
    if typeid==1:   # choose action (s)
        x=np.array([x])
        x=scaler.transform(x)
    else:   # batch normal(s)
        x=np.array(x)
        x = scaler.transform(x)
        x=torch.FloatTensor(x)
    return x

def get_scaler():
    # all data mean and std
    x=np.loadtxt('memory_wc/pool.txt', delimiter=' ')
    x=x[:,:8]  # all s   wc 1310   pr 1225  km 1480   ts 1151
    standardscaler = StandardScaler()
    scaler=standardscaler.fit(x)    # scaler-> mean,var
    return scaler

def PNN_online(dir):
    scaler = get_scaler()
    env = SparkENV()
    s_dim = 8
    a_dim = 32
    deepcat=DeepCAT_PNN(a_dim, s_dim, scaler)
    deepcat.memory=np.loadtxt(current_directory+'memory_ts/pool.txt', delimiter=' ')
    deepcat.load_model(dir)
    time=[]
    deepcat.policy.freeze_preNET() 
    s = env.reset()
    for j in range(10):
        print('s=', s)
        a = deepcat.choose_action(s)
        a = np.clip(np.random.normal(a, 0.1), 0, 1)  # add randomness to action selection for exploration
        print('a=', a)
        s_, r, done, dur = env.step(a, 1, j)
        time.append(dur)
        deepcat.store_transition(s, a, r, s_)
        deepcat.finetune()
        print('reward=', r, '----- dur=', dur)
        s=s_
    print(time)

if __name__=='__main__':
    #step1: choose most similar worklaod based on runing logs
    basemodel_dir=get_similar_workload()
    basemodel_dir=current_directory+basemodel_dir
    #step2: runing with PNN network
    PNN_online(basemodel_dir)
