import numpy as np
import math
from util import a2file,test,getduration,getnextstate,init_state
import numpy as np
class SparkMAENV():
    def __init__(self):
        super(SparkMAENV,self).__init__()
        self.n=3
        self.observation_space=[7,2,2]      # total 11
        # self.action_space=[17,3,4]         total 24
        self.recordmin=20
        self.recordmax=60
        self.start0=30

        # self.s=np.array([0,0,0])

    def isbest(self,dur): # 可优化
        if dur<self.recordmin:
            # self.recordmin=dur
            print('最少时间改变为:',dur)
            return True
        else:
            return False

    def isworse(self,dur):
        if dur>self.recordmax:
            print('超出最大时间！')
            return True
        else:
            return False

    def reset(self):
        s=[]
        sparkobs=np.array([0,0,0.5,0.6,0.5,0.5,0.5])
        #vs t-1,vs t0,parallelism,fraction,exe_num,exe_core,exe_m
        yarnobs=np.array([0.5,0.5])
        #total m,core
        dfsobs=np.array([0.5,0.5])
        #handler_count,bandwith
        s.append(sparkobs)
        s.append(yarnobs)
        s.append(dfsobs)
        return s


    def step(self,a,i,j):
        #把a apply到spark中，得到s'以及相关指标计算r
        #模拟分数--》ml预测分数
        #模拟s'-->
        tmpaction = []
        tmpaction.append(np.concatenate(a[:]))
        action = tmpaction[0]

        a2file(action,i,j)

        #把配置写进 {rep}_app_config.yml文件  yaml.dump 105r
        test(i,j)

        #进行测试 ok
        #文件读出时间 吞吐量等  用哪个？

        #下一时刻的状态  怎么搞
        dur=getduration(i,j)
        # x=input()
        # dur=int(x)

        print('t:',dur)
        dur0=getduration(i,0)
        print('t0:',dur0)
        y = (self.start0 - dur0) / self.start0
        y = (1 + y) ** 2 - 1
        #this trend good or bad
        if j==0:
            durt_1=30
        else:
            durt_1=getduration(i,j-1)
            print('t-1:',durt_1)
        x = (dur - durt_1) / durt_1
       # --------------------reward--------------------------
        # if y <= 0:  # bad trendency
        #     if x < 0:  # good    t down
        #         r = 0
        #     else:  # bad     t up
        #         r = format(y * (1 + x), '.3f')
        # else:  # good trendency
        #     if x > 0:  # bad     t up
        #         r = 0
        #     else:  # good    t down
        #         r = format(y * (1 - x), '.3f')
        # r=0.x
        # ------------------------reward-------------------------
        r=(30-dur)/10
        # r=format(tmp,'.2f')
        s_=[]
        reward=[]
        sparkobs_=[]
        yarnobs_=[]
        dfsobs_=[]
        sparkobs_.append(x)
        sparkobs_.append(y)
        sparkobs_.append(action[0])
        sparkobs_.append(action[9])
        sparkobs_.append(action[12])
        sparkobs_.append(action[13])
        sparkobs_.append(action[14])
        sparklist2array=np.array(sparkobs_)
        yarnobs_.append(action[13])
        yarnobs_.append(action[14])
        yarnlist2array = np.array(yarnobs_)
        dfsobs_.append(action[20])
        dfsobs_.append(action[21])
        dfslist2array = np.array(dfsobs_)
        s_.append(sparklist2array)
        s_.append(yarnlist2array)
        s_.append(dfslist2array)
        # s : with the t0 and t-1 compare
        flag1=self.isbest(dur)
        flag2=self.isworse(dur)

        # r=getreward(i,j)

        if flag1:
            r=2
        if flag2:
            r=-2
        reward=[r for i in range(3)]
        print('reward=',r)
        print('s\'=',s_)
        return s_,reward,dur

# if __name__=='__main__':

#     env=SparkMAENV()
#     x=env.reset()
#     print(x)