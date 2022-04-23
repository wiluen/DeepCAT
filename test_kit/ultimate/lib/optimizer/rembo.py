import numpy as np
from random import random
from .bo import BayesianOptimizer


class RemboOptimizer():
  def __init__(self, config, rembo_conf={}):
    self.D = len(config.items())  #D-dim
    ###############d####################
    self.d = 30  #d-dim
    self.A = np.random.randn(self.D, self.d)  # gen a random matrix A

    self.config = {**config}
    # generate low_dim config, and the config_space
    self.inner_config_space = {}
    for i in range(self.d):
      self.inner_config_space[f'x{i}'] = (-self.d ** 0.5, self.d ** 0.5)

    self.bo = BayesianOptimizer(space=self.inner_config_space, conf=rembo_conf)

  def get_conf(self):
    sample = self.bo.get_conf()  #d-dim
    #print(sample)
    converted = self._convert_back(sample) #D-dim list with [-1,1]
    # first is continuous value, second is translated
    #print(sample)
    #print(converted)
    #print(dict(zip(self.config.keys(), converted)))
    #return bo.X, real configs
    return sample, self._translate(dict(zip(self.config.keys(), converted)))

  def add_observation(self, ob):
    x, y = ob
    #print(x,y)
    self.bo.add_observation((x, y))

  def random_sample(self):
    result = {}
    for k, v in self.config.items():
      v_range = v.get('range')
      if v_range:
        result[k] = random() * len(v_range)
      else:
        minn, maxx = v.get('min'), v.get('max')
        result[k] = random() * (maxx - minn) + minn
    return result, self._translate(result)

  def _convert_back(self, sample):
    # sample is a dict
    # low_dim back to high_dim
    sample = np.array([[v] for v in sample.values()])  # sample to d*1 vector
    converted = list(np.matmul(self.A, sample).flatten())  # D=A * d
    #print(converted)
    to_scale=(-1,1)
    c,d=to_scale
    origin_scale=(min(converted),max(converted))
    a,b=origin_scale

    to_converted=converted
    for k,v in enumerate(converted):
        to_converted[k]*=(d-c)/(b-a)
        to_converted[k]+=c-a*(d-c)/(b-a)

    return to_converted
    #return np.clip(converted, -1, 1).tolist()  # clip the values into [-1, 1]

    ############################
    #sample, D*1 [-1,1] list from _convert_back(d*1)
    #translate from D*1 [-1,1] to origin D*1[min,max]
    ############################
  def _translate(self, sample): # sample=dict(zip(self.config.keys(),converted)),D [-1,1] to D origin
    result = {}
    # orders in sample are the same as in _config dict
    #   see: https://github.com/fmfn/BayesianOptimization/blob/d531dcab1d73729528afbffd9a9c47c067de5880/bayes_opt/target_space.py#L49
    #   self.bounds = np.array(list(pbounds.values()), dtype=np.float)
    #print(dict(zip(sample.values(), self.config.items())))
    for sample_value, (k, v) in zip(sample.values(), self.config.items()):
      #print(sample_value)
      #print(k)
      #print(v)
      v_range = v.get('range')
      if v_range:
        try:
          sample_value = self._rescale(
              sample_value, to_scale=(0, len(v_range))
          )
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
        sample_value = self._rescale(
            sample_value, to_scale=(v.get('min'), v.get('max'))
        )
        result[k] = sample_value if is_float else int(sample_value)
    #print(result)
    #result real configs
    return result

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
