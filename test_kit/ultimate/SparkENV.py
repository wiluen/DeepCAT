import numpy as np
import math
from util import a2file,test,getduration,getnextstate,init_state
class SparkENV():
    def __init__(self):
        super(SparkENV,self).__init__()
        self.n_states=8     # 6*2+2
        self.n_actions=32
        self.recordmin=40
        self.recordmax=200
        self.start0=80  # best :60   60 30 80 150
        self.loadtype=0.87# [0.2ts ,0.4wc ,0.6pr ,0.8km]
        self.datasize=0.56# 3.2g   024g  5.6g

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
        s=init_state(self.loadtype,self.datasize)
        return s


    def step(self,a,i,j):
        a2file(a,i,j)
        #把配置写进 {rep}_app_config.yml文件  yaml.dump 105r
        test(i,j)
        # x=input()
        # dur=int(x)
        dur=getduration(i,j)
        print('duration(s):',dur)

        r=(self.start0-dur)/self.start0
        r=round(r,3)
        # if dur==100:
        #     r=-1
        s_=getnextstate(i,j,self.loadtype,self.datasize)
        flag1=self.isbest(dur)
        done=False

        if flag1:
            r=1
            done=True
        print('reward=',r)
        return s_,r,done,dur

        # dur0=getduration(i,0)
        # print('t0:',dur0)
        # -----------------------cdb reward--------------------------
        # y = (self.start0 - dur0) / self.start0
        # y = (1 + y) ** 2 - 1
        # if j==0:
        #     durt_1=30
        # else:
        #     durt_1=getduration(i,j-1)
        #     print('t-1:',durt_1)
        # x = (dur - durt_1) / durt_1
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