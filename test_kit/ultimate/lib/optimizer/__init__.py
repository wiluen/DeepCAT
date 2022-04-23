from .rand import RandOptimizer
from .bo import ConfigedBayesianOptimizer
from .anneal import AnnealOptimizer
from .grid import GridOptimizer
from .rembo import RemboOptimizer
from .impranking import ImpRank
from .bo_dropout import BOdropoutOptimizer
from .LHSRanking import LHSRank

available_optimizer = [
    'rand',
    'bo',
    'rembo',
    'anneal',
    'grid',
    'imprank',
    'bodropout',
    'lhsrank'
]


def create_optimizer(name, configs, extra_vars={}):
  assert name in available_optimizer, f'optimizer [{name}] not supported.'
  if name == 'rand':
    return RandOptimizer(configs)
  elif name == 'imprank':
    return ImpRank(configs, bucket_num=extra_vars.get('bucket_num'))
  elif name == 'lhsrank':
    return LHSRank(configs, sample_nums=extra_vars.get('sample_nums'))
  elif name == 'bo':
    return ConfigedBayesianOptimizer(configs, bo_conf=extra_vars)
  elif name == 'rembo':
    return RemboOptimizer(configs, rembo_conf=extra_vars)
  elif name == 'bodropout':
    return BOdropoutOptimizer(configs, dropout_conf=extra_vars)
  elif name == 'anneal':
    return AnnealOptimizer(configs)
  elif name == 'grid':
    return GridOptimizer(configs, active_list=extra_vars.get('active'))
