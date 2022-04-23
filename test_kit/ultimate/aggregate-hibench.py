import asyncio
import yaml
import re
import sys
from csv import DictWriter
from pathlib import Path
from statistics import stdev, mean, median
from scipy import stats
from math import nan
from lib import assign, choose_metric_results

######################
# based on zhuchu's version
######################
#####################
# for redis
#####################

def parse_cmd():
  args = sys.argv
  try:
#    assert len(args) > 1, 'too few arguments'
#    _, conf_path, *others = args
    others = ['task_name=spark-bodropout-test', 'out=spark-bodropout-test.csv']
    conf = yaml.load(Path('aggregate/agg_hibench_result.yml').read_text())

    regexp = re.compile(r'(.+)=(.+)')
    for other in others:
      match = regexp.match(other.strip())
      k, v = match.groups()
      assign(conf, k, v)
    return Agg_Config(conf)
  except Exception as e:
    print(e.args)
    print('Usage: python aggregate.py <path_to_conf> [path.to.attr=value]')


class Agg_Config:
  def __init__(self, obj):
    self.target = obj['target']
    self.task_name = obj['task_name']
    self.metric = Metric_Config(obj['metric'])
    self.overwrite = obj['overwrite']
    self.out = obj['out']
    self.tester = obj['tester']


class Metric_Config:
  def __init__(self, obj):
    self.len = obj['len']
    self.mean = obj['mean']
    self.stdev = obj['stdev']
    self.CI_upper = obj['CI_upper']
    self.CI_lower = obj['CI_lower']
    self.CI_r = obj['CI_r']
    self.percent = obj['percent']

    self._obj = obj

  def to_obj(self):
    return self._obj


config = parse_cmd()

assert config is not None

# paths
proj_root = Path(__file__, '../../..').resolve()
result_dir = proj_root / f'target/{config.target}/results/' / config.task_name
out_path = Path.cwd() / 'result' / config.out

# output check
if out_path.exists():
  print(f'{out_path} already exists.')
  if config.overwrite:
    out_path.unlink()
    print('deleted.')
  else:
    print('aborted.')
    sys.exit()



# ycsb, return float value
def get_hibench_duration(content):
  regexp = re.compile(re.compile(
      r'Duration\(s\), (\d+(?:\.\d+)?)'))
  match = regexp.search(content)
  assert match is not None, f'result not match, content: {content}'
  return float(match.group(1))


#result parser name
extractor = {
    'hibench': {
        'ex': get_hibench_duration,
    },
}

assert config.tester in extractor, f'extractor for {config.tester} not exist'

extract = extractor[config.tester]

fieldnames = [f'{key}_{metric}' for key in extract.keys()
              for metric, flag in config.metric.to_obj().items() if flag is True]
writer = DictWriter(open(out_path, 'w'), fieldnames=['task_id', *fieldnames],
                    delimiter=',', lineterminator='\n')
writer.writeheader()

all_result = {}
regex = re.compile(r'(\d+)_run_result_(\d+)')
re_type = type(regex)

for p in result_dir.iterdir():  # pylint: disable=E1101
  if p.is_file():
    match = regex.match(p.name)
    if match:
      task_id, rep_num = match.groups()
      if task_id not in all_result:
        all_result[task_id] = []

      # parse content
      content = p.read_text()
      result = {}
      for k, pattern in extract.items():
        if type(pattern) is re_type:
          match = pattern.search(content)  # pylint: disable=E1101
          if match is not None:
            result[k] = match.group(1)
        elif callable(pattern):
          result[k] = pattern(content)
      if len(result) == len(extract):  # all matched
        all_result[task_id].append(result)

# calculate medians
medians = {}
if config.metric.percent:
  for k in extract.keys():
    all_values = [float(a_result[k]) for id_results in all_result.values()
                  for a_result in id_results]
    medians[k] = median(all_values) if len(all_values) != 0 else nan


for task_id, rep_results in all_result.items():
  id_result = {}
  for k in extract.keys():
#    values = [float(result[k]) for result in rep_results]
#    # filter out invalid results. e.g throughput = 0.0
#    values = [v for v in values if v > 0.0]

    #new
    old_values = [float(result[k]) for result in rep_results]
    values = choose_metric_results(old_values)
    values = [v for v in values if v > 0.0]

    count = len(values)
    _mean = mean(values) if len(values) != 0 else nan
    _stdev = stdev(values) if len(values) > 1 else 0.0
    confidence_level = .95
    # t-value for inverse_t_distribution
    t = stats.t.ppf(1 - (1 - confidence_level) / 2, count - 1)
    upper_CI = _mean + t * _stdev / count ** 0.5
    lower_CI = _mean - t * _stdev / count ** 0.5
    r_CI = (upper_CI - lower_CI) / 2 / _mean

    if config.metric.len:
      id_result[f'{k}_len'] = count
    if config.metric.mean:
      id_result[f'{k}_mean'] = _mean
    if config.metric.stdev:
      id_result[f'{k}_stdev'] = _stdev
    if config.metric.CI_upper:
      id_result[f'{k}_CI_upper'] = upper_CI
    if config.metric.CI_lower:
      id_result[f'{k}_CI_lower'] = lower_CI
    if config.metric.CI_r:
      id_result[f'{k}_CI_r'] = r_CI
    if config.metric.percent:
      if k not in medians:
        # medians[k] = median(values)
        medians[k]
      id_result[f'{k}_percent'] = _mean / medians[k]

  writer.writerow({
      'task_id': task_id,
      **id_result
  })

print('done.')
