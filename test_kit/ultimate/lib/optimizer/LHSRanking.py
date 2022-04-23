import numpy as np
from .base import BaseOptimizer
#from smt.sampling_methods import LHS

class LHSRank(BaseOptimizer):
  def __init__(self, para_setting, sample_nums):
    super().__init__(para_setting)
    #bucket_num to
    self.sample_nums = sample_nums
    #convert dict to list
    keys=self.para_setting.keys()
    vals=self.para_setting.values()
    self.configList=[(key,val) for key,val in zip(keys,vals)]
    # print('configList:')
    # print(self.configList)

    #to generate a LHS array (D,2),current D=41
    xlimits = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    sampling = LHS(xlimits=xlimits)
    self.lhsvalues = sampling(self.sample_nums)
    self.lhsindex = 0

  def get_conf(self):
    #return (config, value) dict
    assert self.lhsindex <= self.sample_nums, f'sample_nums must be carefully be set =iter_limit'

    lhsarray = self.lhsvalues[self.lhsindex,:]
    # print('Current index and the corresponding lhsarray')
    # print (self.lhsindex)
    # print (lhsarray)

    #obtain (config, default value) dict
    result={}
    for k, conf in self.para_setting.items():
        result[k]= conf.get('default')
    #print (result)

    #modify default value with the lhsarray
    result_lhs=dict(zip(result.keys(), lhsarray))
    # print ('show configs with lhsarray:')
    # print (result_lhs)

    #obtain the configs
    for key in result_lhs.keys():
        value = result_lhs.get(key)
        value_range = self.para_setting.get(key).get('range')
        if value_range:
            tempV = self._rescale(origin_v=result_lhs[key], to_scale=(0,len(value_range)))
            result_lhs[key] = tempV
        else:
            tempV = self._rescale(origin_v=result_lhs[key], to_scale=(self.para_setting.get(key).get('min'),self.para_setting.get(key).get('max')))
            result_lhs[key] = tempV
    # print('show result_lhs:')
    # print(result_lhs)
    lhsconfig=self._translate(result_lhs)
    # print('show result_lhs configs:')
    # print(lhsconfig)
    #use next lhsarray
    self.lhsindex = self.lhsindex+1
    # #note that configd is d-dimenal , self...is D-Config
    #return configd, self._translate(random_configD)
    return result_lhs, lhsconfig
  # means no learning, just search
  def add_observation(self, ob):
    pass

 #scale from (0,1) to the original scale, but not the ture configurations
  def _rescale(self, origin_v, to_scale, origin_scale=(-1, 1)):
   a, b = origin_scale
   c, d = to_scale
   if origin_v > b:
      origin_v = b
   if origin_v < a:
      origin_v = a
   to_v=origin_v
   to_v *= (d - c) / (b - a)  # scale
   to_v += c - a * (d - c) / (b - a)  # offset
   return to_v

 #from numeric valuse to configs
  def _translate(self, sample):
    result = {}
    # orders in sample are the same as in _config dict
    #   see: https://github.com/fmfn/BayesianOptimization/blob/d531dcab1d73729528afbffd9a9c47c067de5880/bayes_opt/target_space.py#L49
    #   self.bounds = np.array(list(pbounds.values()), dtype=np.float)
    for sample_value, (k, v) in zip(sample.values(), self.para_setting.items()):
      v_range = v.get('range')
      if v_range:
        try:
          index = int(sample_value)
          if index == len(v_range):
            index -= 1
          result[k] = v_range[index]
        except Exception as e:
          print('ERROR!')
          print(k, sample_value)
          print(v_range)
          raise e
      else:
        is_float = v.get('float', False)
        result[k] = sample_value if is_float else int(sample_value)
    #print(result)
    return result
