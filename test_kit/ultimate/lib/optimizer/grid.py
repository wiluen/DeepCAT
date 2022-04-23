from .base import BaseOptimizer
from math import floor


class GridOptimizer(BaseOptimizer):
  def __init__(self, para_setting, active_list):
    super().__init__(para_setting)
    self.active_list = active_list
    #print(active_list)

    for name in active_list:
      assert name in para_setting, f'{name} not in parameters'
    self.base = dict([[k, self.para_setting[k]['default']]
                      for k in self.para_setting if k not in self.active_list])
    #print(self.base)
    self.it = self._createIterator()

  def get_conf(self):
    #print(next(self.it))
    return None, next(self.it)

  def _createIterator(self):
    # yield from self._iter(0)
    for conf in self._iter(0):
      yield {**self.base, **conf}

  def _iter(self, L, prev={}):
    if L >= len(self.active_list):
      yield {**prev}
      return
    name = self.active_list[L]
    conf = self.para_setting[name]
    ##################conf for active configs
    _range = conf.get('range')
    #print(conf)
    if _range is None:
      bucket_num = conf.get('bucket_num')
      assert bucket_num is not None, f'bucket_num is undefined in {name}'
      assert bucket_num > 1, f'bucket_num must be greater than 1'
      minn = conf.get('min')
      maxx = conf.get('max')
      allow_float = conf.get('float', False)
      if not allow_float:
        slots = maxx - minn + 1
        assert slots >= bucket_num, f'bucket_num (f{bucket_num}) too large for f{name}'
        # bucket_size = (maxx - minn) / bucket_num
        # for i in range(bucket_num):
        #   v = int(floor(minn + bucket_size * (i + .5)))
        #   yield from self._iter(L + 1, {**prev, name: v})
        step_size = (maxx - minn) / (bucket_num - 1)
        for i in range(bucket_num):
          v = int(round(minn + step_size * i))
          yield from self._iter(L + 1, {**prev, name: v})
      else:
        step_size = (maxx - minn) / (bucket_num - 1)
        for i in range(bucket_num):
          yield from self._iter(L + 1, {**prev, name: minn + step_size * i})
    else:
      for item in _range:
        yield from self._iter(L + 1, {**prev, name: item})

  def add_observation(self, ob):
    pass

  def dump_state(self, path):
    pass
