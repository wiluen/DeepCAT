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
    result = {}
    for k, conf in self.para_setting.items():
      result[k] = conf.get('default')

      # from task_id and bucket_num to obtain the index in configList
    firstIndex = 0
    secondIndex = 1
    firstflag = (task_id - 1) // self.bucket_num
    secondflag = (task_id - 1) % self.bucket_num

    result = self.get_twoConf(firstflag, secondflag, firstIndex, secondIndex)
    return result

  def get_twoConf(self, firstflag, secondflag, firstIndex, secondIndex):
    # get key of confIndex
    config_name_first = self.configList[firstIndex][0]
    # get content of confIndex, it is a dict
    config_content_first = self.configList[firstIndex][1]

    config_name_second = self.configList[secondIndex][0]
    config_content_second = self.configList[secondIndex][1]

    # get content from config_content

    # if firstflag< self.bucket_num
    minn_first = config_content_first.get('min')
    maxx_first = config_content_first.get('max')
    range_first = config_content_first.get('range')

    minn_second = config_content_second.get('min')
    maxx_second = config_content_second.get('max')
    range_second = config_content_second.get('range')

    result = {}


    if range_first is None:
      step_size_first = (maxx_first - minn_first) / (self.bucket_num - 1)
      result[config_name_first] = int(round(minn_first + step_size_first * firstflag))
      print(config_name_first, ":", result[config_name_first])
    else:
      # for item in _range:
      canditNum = len(range_first)
      minNum = 0
      if minNum + firstflag <= canditNum - 1:
        result[config_name_first] = range_first[minNum + firstflag]
      else:
        result[config_name_first] = range_first[canditNum - 1]
      print(config_name_first, ":", result[config_name_first])

    if range_second is None:
      step_size_second = (maxx_second - minn_second) / (self.bucket_num - 1)
      result[config_name_second] = int(round(minn_second + step_size_second * secondflag))
      print(config_name_second, ":", result[config_name_second])
    else:
      canditNum = len(range_second)
      minNum = 0
      if minNum + secondflag <= canditNum - 1:
        result[config_name_second] = range_second[minNum + secondflag]
      else:
        result[config_name_second] = range_second[canditNum - 1]
      print(config_name_second, ":", result[config_name_second])
    
    return result

  # means no learning, just search
  def add_observation(self, ob):
    pass
