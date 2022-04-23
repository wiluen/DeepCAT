from random import random, randint, choice

from .base import BaseOptimizer
from ..other import random_sample, get_default

from math import floor


#from grid and rand to write codes fro ranking importance

class ImpRank(BaseOptimizer):
  def __init__(self, para_setting, bucket_num):
    super().__init__(para_setting)
    #bucket_num to
    self.bucket_num = bucket_num
    #convert dict to list
    keys=self.para_setting.keys()
    vals=self.para_setting.values()
    self.configList=[(key,val) for key,val in zip(keys,vals)]
    #print(self.configList)

  def get_conf(self, task_id):
    #return (config, value) dict
    result={}
    for k, conf in self.para_setting.items():
        result[k]= conf.get('default')
    #print (result)

    #from task_id and bucket_num to obtain the index in configList
    confIndex=int((task_id-1)/self.bucket_num)
    maxIndex=len(self.configList)
    #print ("maxIndex:", maxIndex)
    assert confIndex <= maxIndex-1, f'taks_id must be carefully be set =maxIndex*bucket_num'
    #get key of confIndex
    config_name = self.configList[confIndex][0]
    #get content of confIndex, it is a dict
    config_content = self.configList[confIndex][1]

    #get content from config_content
    minn = config_content.get('min')
    maxx = config_content.get('max')
    _range = config_content.get('range')

    #is numeric variables
    if _range is None:
        allow_float = config_content.get('float', False)
        if not allow_float: #int
            slots = maxx - minn + 1
            if slots >= self.bucket_num:
                step_size = (maxx-minn) / (self.bucket_num-1)
                i=(task_id-1) % self.bucket_num
                result[config_name]=int(round(minn+step_size*i))
                print(config_name, ":" ,result[config_name])
                return result
            else: #like(0,1), need repeat
                i=(task_id-1) % self.bucket_num
                if minn+i <= maxx:
                    result[config_name]=minn+i
                else:
                    result[config_name]=maxx
                print(config_name, ":" ,result[config_name])
                return result
        else:#float
                step_size = (maxx-minn) / (self.bucket_num-1)
                i=(task_id-1) % self.bucket_num
                result[config_name]=minn+step_size*i
                print(config_name, ":" ,result[config_name])
                return result
    #is category variables
    else:
        # for item in _range:
        canditNum = len(_range)
        minNum=0
        i=(task_id-1) % self.bucket_num
        if minNum+i <= canditNum-1:
            result[config_name]=_range[minNum+i]
        else:
            result[config_name]=_range[canditNum-1]
        print(config_name, ":" ,result[config_name])
        return result

  # means no learning, just search
  def add_observation(self, ob):
    pass
