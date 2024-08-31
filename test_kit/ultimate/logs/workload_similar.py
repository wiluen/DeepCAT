import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist

model_dict={'wc.txt':['model/wc/Actor_td3.pt','model/wc/Critic1_td3.pt','model/wc/Critic2_td3.pt'],
            'ts.txt':['model/ts/Actor_td3.pt','model/ts/Critic1_td3.pt','model/ts/Critic2_td3.pt'],
            'pr.txt':['model/pr/Actor_td3.pt','model/pr/Critic1_td3.pt','model/pr/Critic2_td3.pt'],
            'km.txt':['model/km/Actor_td3.pt','model/km/Critic1_td3.pt','model/km/Critic2_td3.pt']
            }
Alog_feature=[]
target_worklaod_feature=[]
exist_workload_feature=[]
similarity=[]
file=['wc.txt','ts.txt','pr.txt','km.txt']
with open('agg.txt') as f:
    seq = f.read()
    for i in range(0,16):
        character_i = str(seq.split('\n')[i])
        Alog_feature.append(float(character_i.split(':')[1]))
target_worklaod_feature.append(Alog_feature)

for filename in file:
    with open(filename) as f:
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
print('base model:',model_dict[file[min_index]])
