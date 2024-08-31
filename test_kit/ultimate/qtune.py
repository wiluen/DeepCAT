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

'''
implement of G. Li, X. Zhou, S. Li, and B. Gao: Qtune: A query-aware database tuning system with deep reinforcement learning Proceedings of the VLDB Endowment,2019
'''
#####################  hyper parameters  ####################

MAX_EPISODES = 50
MAX_EP_STEPS = 10
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 1151
BATCH_SIZE = 32
directory="../model_qtune/"



###############################  DDPG  ####################################

class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, a_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(a_dim)
        self.out.weight.data.normal_(0, 0.1)

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


class MLP(nn.Module):
    def __init__(self,n_feature,n_deltaS):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(n_feature,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,n_deltaS)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.tanh(self.fc3(x))
        return x



class Predictor(object):
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 32
        self.n_feature=6
        self.n_deltaS=6
        self.predictor=MLP(self.n_feature,self.n_deltaS)
        self.optimizer=torch.optim.Adam(self.predictor.parameters(),lr=LR_A)
        self.lossP=nn.MSELoss()
        self.memory=[]

    def train(self):
        x = np.loadtxt('memory-qtune/wordcount_history', delimiter=' ')
        x = x[:, :6]  # all s   wc 1310   pr 1225  km 1480   ts 1151
        standardscaler = StandardScaler()
        scaler = standardscaler.fit(x)

        losslist=[]
        for i in range(4000):
            index=np.random.choice(1600,BATCH_SIZE)
            memory=self.memory[index,:]
            state=torch.FloatTensor(memory[:,6:12])
            next_state=torch.FloatTensor(memory[:,-7:-1])
            feature=torch.FloatTensor(memory[:,:6])

            feature = scaler.transform(feature)
            feature = torch.FloatTensor(feature)

            g=self.predictor(feature)
            deltaS=next_state-state
            loss=self.lossP(g,deltaS)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losslist.append(loss.item())
        torch.save(self.predictor, directory + "predictor.pt")

class DDPG(object):
    def __init__(self, a_dim, s_dim,scaler):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        # self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.scaler = scaler

    def choose_action(self, s):  # s []
        s = normalization(s, 1, self.scaler)  # s [[]] for normalize
        s = torch.FloatTensor(s)
        self.Actor_eval.eval()  # when BN or Dropout in testing ################
        # s = torch.unsqueeze(torch.FloatTensor(s), 0)
        act = self.Actor_eval(s)[0].detach()  # ae（s）
        self.Actor_eval.train()  ###############
        return act


    def learn(self):
        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        #
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')
        for (target_param,param) in zip(self.Actor_target.parameters(),self.Actor_eval.parameters()):
            target_param.data.copy_(
                target_param.data*(1-TAU)+param.data*TAU
            )
        for (target_param,param) in zip(self.Critic_target.parameters(),self.Critic_eval.parameters()):
            target_param.data.copy_(
                target_param.data*(1-TAU)+param.data*TAU
            )

        indices = np.random.choice(self.pointer, size=BATCH_SIZE)
        # indices=[]
        # for i in range(303):
        #     indices.append(i)

        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim+1])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim-1:-1])
        bd = torch.FloatTensor(bt[:, -1:])
        #
        bs = normalization(bs, 0, self.scaler)
        bs_ = normalization(bs_, 0, self.scaler)

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)

        loss_a = -torch.mean(q)

        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + (1-bd)*GAMMA * q_  # q_target = 负的
        q_v = self.Critic_eval(bs, ba)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print('critic td_error=',td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

#[84,120,288,372,522,838,1368,2164,3388,5099]

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        # self.memory[self.pointer, :] = transition
        self.pointer += 1


    def load_model(self, episode):
        model_name_c = "Critic_cdbtune_" + str(episode) + ".pt"
        model_name_a = "Actor_cdbtune_" + str(episode) + ".pt"
        self.Critic_target = torch.load(directory + model_name_c)
        self.Critic_eval = torch.load(directory + model_name_c)
        self.Actor_target = torch.load(directory + model_name_a)
        self.Actor_eval= torch.load(directory + model_name_a)

    def save_model(self, episode):
        model_name_c = "Critic_cdbtune_" + str(episode) + ".pt"
        model_name_a = "Actor_cdbtune_" + str(episode) + ".pt"
        torch.save(self.Critic_eval, directory + model_name_c)
        torch.save(self.Actor_eval, directory + model_name_a)
###############################  training  ####################################


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
    x=np.loadtxt('memory_ts/pool_newstate.txt', delimiter=' ')
    x=x[:,:8]  # all s   wc 1310   pr 1225  km 1480   ts 1151
    standardscaler = StandardScaler()
    scaler=standardscaler.fit(x)    # scaler-> mean,var
    print("scaler:pr all sample s:8v")
    print('每列的均值', scaler.mean_)
    
    print('每列的方差', scaler.scale_)
    return scaler


def collect_data(x):
    scaler=get_scaler()
    env = SparkENV()
    s_dim = 8
    a_dim = 32
    count = 0
    ddpg = DDPG(a_dim, s_dim,scaler)
    ddpg.load_model(x)
    print(f'============load model {x}iters===============')
    # ddpg.memory = np.loadtxt('pool_wordcount_32v_3.txt', delimiter=' ')
    # ddpg.pointer = 600
    print('============load memory===============')
    y = []
    alltime = []
    var = 0.05  # control exploration
    # for i in range(8):
    s = env.reset()
    ep_reward = 0
    # for i in range(10):
    for j in range(300):
        print('s=',s)
        starttime = datetime.datetime.now()
        count+=1
        a = ddpg.choose_action(s)
        end = datetime.datetime.now()
        print(end-starttime)
        
        s_, r, done, dur = env.step(a,1,j)

        print('reward=',r,'----- dur=',dur)

        alltime.append(dur)

        if done == True:
            done=1
        else:
            done=0
        ddpg.store_transition(s, a, r, s_, done)
        s = s_
        print('--------------------------------------')
        



def offline_train(x):            #x:history model
    scaler = get_scaler()
    env = SparkENV()
    s_dim = 8
    a_dim = 32
    ddpg = DDPG(a_dim, s_dim,scaler)
    # ddpg.load_model(x)
    print(f'============load model {x}iters===============')
    ddpg.memory = np.loadtxt('memory_wc/pool_newstate.txt', delimiter=' ')
    ddpg.pointer = 1310
    print('============load memory===============')
    for i in range(10002):
        # for j in range(5):     #equal with td3
        ddpg.learn()
        if i>1 and i%500==0:
            print(f'train {i} iter')
            ddpg.save_model(i)




def train(x):
    VAR=0.5
    scaler = get_scaler()
    env = SparkENV()
    s_dim = 8
    a_dim = 32
    count=x   #save model
    ddpg = DDPG(a_dim, s_dim, scaler)
    ddpg.load_model(x)
    print(f'============load model {x}iters===============')
    ddpg.memory = np.loadtxt('memory_ts/pool_newstate.txt', delimiter=' ')
    ddpg.pointer = 1396 # 1151+245
    print('============load memory===============')
    for i in range(200):
        s=env.reset()
        ep_reward=0
        for j in range(5): #5 step
            a=ddpg.choose_action(s)
            # a = np.clip(np.random.normal(a, VAR), 0.1, 0.95)
            a = np.clip(a, 0.2, 0.8)
            s_, r, done, dur = env.step(a, i, j)
            # VAR *= 0.95
            # print('VAR=',VAR)
            # s_=s
            print(ddpg.pointer)
            # r=1
            # done=False
            # dur=30
            if done == True:
                done=1
            else:
                done=0
            ddpg.store_transition(s, a, r, s_, done)
            for k in range(5):  #5'learn-1'test
                ddpg.learn()
            s=s_
            ep_reward+=r
        print('ep_reward=',ep_reward)
        count+=1
        if i%6==0:
            np.savetxt(f'offline_ts_time/pool_ts_cdbtune.txt',ddpg.memory,fmt='%f',delimiter=' ')
            ddpg.save_model(count)
            print('memory pool save..')

def Qtune():
    VAR=3
    totol_r=0
    env=SparkENV.ENV()
    y=[]
    ddpg=DDPG()
    P=Predictor()
    for i in range(200):
        s = env.reset()

        ep_reward = 0
        for j in range(10):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), 0, 1)    # add randomness to action selection for exploration
            s_, r, done, dur,feature= env.Qtunestep(a,i,j)
            ddpg.store_transition(s, a, r , s_,feature)
            deltaS=ddpg.predictor(feature)
            s_+=deltaS
            s = s_
            ep_reward += r
            if j == STEP-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                print(a)
              
                y.append(f)
                # if ep_reward > -300:RENDER = True
                break
    plt.plot(np.arange(200),y)
    plt.xlabel('Episode')
    plt.ylabel('function value')

    plt.show()

def trainPredictor():
    p=Predictor()
    p.memory=np.loadtxt('memory-qtune/wordcount_history', delimiter=' ')
    p.train()
# [390,363,346,423,492]
if __name__=='__main__':
    # offline_train(4000)
    # collect_data(40000)
    # trainPredictor()
    Qtune()
