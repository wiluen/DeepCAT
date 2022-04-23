from ..bayes_opt import BayesianOptimization
from ..bayes_opt.helpers import acq_max, UtilityFunction
from random import sample, randint, random, choice
from .bo import BayesianOptimizer

from ..other import random_sample

#bo with dropout, rewrite based on bo.py
class BOdropoutOptimizer():
  def __init__(self, config, dropout_conf={}):
    self._config = {**config}
    #print(self._config)

    #read the config for dropout like d and fill strategy
    self.dropout_conf=dropout_conf

    #to generate a d-dimesional BO with space [-1,1] for all d
    self.bod_space = {}
    for i in range(self.dropout_conf.get('dropout_d',10)):
        self.bod_space[f'x{i}']=(-1,1)
    print('show bod_space, a d-dimesional space[-1,1]:')
    print(self.bod_space)

    #########conf contains acq, use; else use default
    self.bo_conf={}
    self.bo_conf['acq'] = self.dropout_conf.get('acq', 'ucb')
    self.bo_conf['kappa'] = self.dropout_conf.get('kappa', 2.576)
    self.bo_conf['xi'] = self.dropout_conf.get('xi', 0.0)

    #generate d-dimensional bo optimizer
    self.bod = BayesianOptimizer(space=self.bod_space, conf=self.bo_conf)

  # add new point (xd, y) to the bod, need transform
  def add_observation(self, ob):
    x, y = ob
    #print(x,y)
    self.bod.add_observation((x, y))

  # get conf and convert to legal config
  def get_conf(self, current_ob_x, current_ob_y):
    configd = self.bod.get_conf()
    print('show configd from bod:')
    print(configd)
    # first is continuous value, second is translated

    available_fill = [
    'rand',
    'copy',
    'mix'
    ]
    dropout_fill=self.dropout_conf.get('dropout_fill','rand')
    assert dropout_fill in available_fill, f'fill strategy not supported.'

    dropout_mix_p=self.dropout_conf.get('dropout_mix_p', 0.5)

    #mix, 0 for rand and 1 for copy
    if dropout_fill == 'mix':
        x=random()
        if x < dropout_mix_p:
            dropout_fill = 'copy'
        else:
            dropout_fill = 'rand'
        print('show dropout_fill')
        print(dropout_fill)

    if dropout_fill == 'rand':
        # use d original values and D-d random values
        #generate a random config for D-dimensional
        random_configD,random_configD_trans = self.random_sample()
        print('show random_configD based on xD_max from boD:')
        print(random_configD)
        #use random.sample to obtain d keys need to be modified from configD
        keys_d=sample(list(random_configD),self.dropout_conf.get('dropout_d',10))
        #set the value of keys_d, need scale (-1ï¼?) to the original scale
        print('show the chosen random d keys to modify:')
        print(keys_d)

        configd_i=0
        for key in keys_d:
            value = self._config.get(key)
            value_range = value.get('range')
            if value_range:
                tempV = self._rescale(origin_v=configd[f'x{configd_i}'], to_scale=(0,len(value_range)))
                random_configD[key]= tempV
            else:
                tempV = self._rescale(origin_v=configd[f'x{configd_i}'], to_scale=(value.get('min'),value.get('max')))
                random_configD[key]= tempV
            configd_i=configd_i+1

        print('show random_configD after modifying d keys:')
        print(random_configD)
        #note that configd is d-dimenal , self...is D-Config
        return configd, self._translate(random_configD)

    elif dropout_fill == 'copy':
        index_max = current_ob_y.index(max(current_ob_y))
        current_best_configD_for_copy = current_ob_x[index_max]
        current_best_configD = current_best_configD_for_copy.copy()
        print('show current_best_y:')
        print(max(current_ob_y))
        print('show current_best_configD:')
        print(current_best_configD)

        #use random.sample to obtain d keys need to be modified from configD
        keys_d=sample(list(current_best_configD),self.dropout_conf.get('dropout_d',10))
        #set the value of keys_d, need scale (-1ï¼?) to the original scale
        print('show the chosen random d keys to modify:')
        print(keys_d)

        configd_i=0
        for key in keys_d:
            value = self._config.get(key)
            value_range = value.get('range')
            if value_range:
                tempV = self._rescale(origin_v=configd[f'x{configd_i}'], to_scale=(0,len(value_range)))
                current_best_configD[key]= tempV
            else:
                tempV = self._rescale(origin_v=configd[f'x{configd_i}'], to_scale=(value.get('min'),value.get('max')))
                current_best_configD[key]= tempV
            configd_i=configd_i+1
        print('show current_best_configD after modifying d keys:')
        print(current_best_configD)
        #note that configd is d-dimenal , self...is D-Config
        return configd, self._translate(current_best_configD)

  #scale from (-1,1) to the original scale, but not the ture configurations
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

  def random_sample(self):
    result = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:
        result[k] = random() * len(v_range)
      else:
        minn, maxx = v.get('min'), v.get('max')
        result[k] = random() * (maxx - minn) + minn
    return result, self._translate(result)

  def _translate(self, sample):
    result = {}
    # orders in sample are the same as in _config dict
    #   see: https://github.com/fmfn/BayesianOptimization/blob/d531dcab1d73729528afbffd9a9c47c067de5880/bayes_opt/target_space.py#L49
    #   self.bounds = np.array(list(pbounds.values()), dtype=np.float)
    for sample_value, (k, v) in zip(sample.values(), self._config.items()):
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
