import numpy as np
import math
import pandas as pd
from util import lauch_tester
file='/home/zyx/workspace/testFlink/results/cdbtune-1500-latency/'

class SparkENV():
    def __init__(self):
        super(SparkENV,self).__init__()
        self.n_states=25     # 6*2+2
        self.n_actions=16


    def reset(self):
        s=[0.17 ,0.35, 0.49, 0.64 ,0.38, 0.4, 0.12, 0.22, 0.3, 0.2, 0.26, 0.31, 0.16 ,0.2, 0.26]
        return s

    def step_flink(self,a,stepid):
        lauch_tester(a,stepid)
        tp,p99=self.get_result(stepid)
        print('============================throughput:',tp,'p99latency:',p99)

        r_tp=(tp-10000)/10000
        r_tp=round(r_tp,3)

        r_p99=(100000-p99)/100000
        r_p99=round(r_p99,3)
        # if dur==100:
        #     r=-1
        s_=self.getnextstate(stepid)
        print('next state:',s_)
        # s_=0
        return s_,r_p99,tp,p99
    
    def get_result(self,stepid):
        res='_run_result.csv'
        file_res=file+str(stepid)+res
        df = pd.read_csv(file_res)
        tp = df['throughput(msgs/s)'][0]
        p99 = df['p99_latency(ms)'][0]
        return tp,p99

    
    def getnextstate(self,stepid):
        state1='_state_slave1.txt'
        state2='_state_slave2.txt'
        state3='_state_slave3.txt'
        state4='_state_slave4.txt'
        state5='_state_slave5.txt'
        
        file_state1=file+str(stepid)+state1
        file_state2=file+str(stepid)+state2
        file_state3=file+str(stepid)+state3
        file_state4=file+str(stepid)+state4
        file_state5=file+str(stepid)+state5
        nextstate=[]
        for f in [file_state1,file_state2,file_state3,file_state4,file_state5]:
            with open(f, 'r') as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            nextstate.append(float(lines[0].split(',')[0]))
            nextstate.append(float(lines[1].split(',')[0]))
            nextstate.append(float(lines[2]))
            # nextstate.append(float(lines[4])/float(lines[3]))
            # nextstate.append(int(lines[5].split(" ")[-2].split("%")[0]))
        return nextstate


    
    
