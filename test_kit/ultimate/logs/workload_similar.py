import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import os
import math

Alog_feature=[]
A=[]
All_feature=[]
similarity=[]
file=['wc.txt','ts.txt','pr.txt','km.txt']
with open('pca.txt') as f:
    seq = f.read()
    for i in range(0,16):
        character_i = str(seq.split('\n')[i])
        Alog_feature.append(float(character_i.split(':')[1]))
A.append(Alog_feature)

for filename in file:
    with open(filename) as f:
        base_feature = []
        seq = f.read()
        for i in range(0, 16):
            character_i = str(seq.split('\n')[i])
            base_feature.append(float(character_i.split(':')[1]))
        All_feature.append(base_feature)

print(A)
print(All_feature)

standardE =cdist(All_feature, A, metric='seuclidean')    #seuclidean
print(standardE)
