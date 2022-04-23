from random import random, randint, choice

from .base import BaseOptimizer
from ..other import random_sample, get_default


class RandOptimizer(BaseOptimizer):
  def __init__(self, para_setting):
    super().__init__(para_setting)

  def get_conf(self):
    return None, random_sample(self.para_setting)

  def add_observation(self, ob):
    pass
