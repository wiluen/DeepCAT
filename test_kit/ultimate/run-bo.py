import asyncio
import random

import yaml
import re
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from statistics import mean
import math
from lib.other import parse_cmd, run_playbook, get_default, choose_metric_results
from lib.optimizer import create_optimizer
from lib.result_parser import parse_result


def find_exist_task_result():
  task_id = -1
  regexp = re.compile(r'(\d+)_.+')
  if result_dir.exists():
    for p in result_dir.iterdir():
      if p.is_file():
        res = regexp.match(p.name)
        if res:
          task_id = max(task_id, int(res.group(1)))
  return None if task_id == -1 else task_id

def translate_config_to_numeric(sample_config, app_setting):
    config = dict(app_setting)
    # default configs, need to transform category into values
    sample_config_v = {}
    for k, v in sample_config.items():
        v_range = config[k].get('range')
        if v_range:
            sample_config_v[k] = v_range.index(v)
        else:
            sample_config_v[k] = v
    return sample_config_v

def divide_config(sampled_config):
  for k in sampled_config.keys():
    if type(sampled_config[k]) is bool:
      # make sure no uppercase 'True/False' literal in result
      sampled_config[k] = str(sampled_config[k]).lower()
    elif type(sampled_config[k]) is np.float64:
      sampled_config[k] = float(sampled_config[k])
  spark_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in spark_setting)
  )
  yarn_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in yarn_setting)
  )
  dfs_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in dfs_setting)
  )
  return spark_config,yarn_config,dfs_config

def _print(msg):
  print(f'[{datetime.now()}]  - {msg}')
  # print('[' + datetime.now() + ']')


async def main(test_config, init_id, app_setting):
  # create optimizer
  optimizer = create_optimizer(
      'bo',
      {
          **app_setting
      },
      extra_vars=test_config.optimizer.extra_vars
  )
  # if hasattr(optimizer, 'set_status_file'):
  #   optimizer.set_status_file(result_dir / 'optimizer_status')

  task_id = init_id
  rep=0
  _print('all start')
  while task_id < 10:
    task_id += 1
    # - sample config
    if task_id == 0:  # use default config 第一轮默认配置
      sampled_config_numeric, sampled_config = None, get_default(app_setting)
      sampled_config_numeric = translate_config_to_numeric(sampled_config, app_setting)   # for GP  numeric
    else: # 贝叶斯优化结果-----------------------------------------------------------------------
      try:
        sampled_config_numeric, sampled_config = optimizer.get_conf()
      except StopIteration:
        # all configuration emitted
        return
    print('sampled_config_numeric=',sampled_config_numeric)           # for GP
    print('sampled_config=',sampled_config)                   # for configuration
    # - divide sampled config app & os 分出os和app配置
    spark_conf,yarn_conf,dfs_conf = divide_config(sampled_config)
    # if tune_app is off, just give sample_app_config a default value这轮不调，给默认值
    # todo : this is OK
    spark_config_path = result_dir / f'{task_id}_{rep}_spark_config.yml'
    yarn_config_path = result_dir / f'{task_id}_{rep}_yarn_config.yml'
    dfs_config_path = result_dir / f'{task_id}_{rep}_dfs_config.yml'
    spark_config_path.write_text(
        yaml.dump(spark_conf, default_flow_style=False)
    )
    yarn_config_path.write_text(
        yaml.dump(yarn_conf, default_flow_style=False)
    )
    dfs_config_path.write_text(
        yaml.dump(dfs_conf, default_flow_style=False)
    )
    _print(f'{task_id}: spark/yarn/dfs config generated..')
    print(sampled_config)
          # run 8 times
    await single_test(
          task_id=task_id,
          rep=rep,
          clients=clients,
    )
# # 结果在_run_result_  ？？怎么放进来的？？
#       # after test, collect metrics for evaluation
    _print(f'{task_id} : parsing result...')
    result = parse_result(
        tester_name='hibench',
        result_dir=result_dir,
        task_id=task_id,
        rep=rep,
        printer=_print
    )
    # result=random.randint(0,9)
    print('result=',result)
      #  The first numerical error may be too large to be added
      # if result is not None and rep != 0 and result != 0.:
              #min time   so :-


    # choose the right metric_results
    # metric_results = choose_metric_results(metric_results_list)


    # after 结果加入优化器 继续优化
    if task_id != 0:  # not adding default info, 'cause default cannot convert to numeric form
      # metric_result = mean(metric_results) if len(metric_results) > 0 else .0
      optimizer.add_observation(
          (sampled_config_numeric, -result)
      )
      print('added observation...')
      if hasattr(optimizer, 'dump_state'):
        optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')
    print('iter done...')
  # after reaching iter limit

  _print('all end')
  global proj_root

  # cleanup things
  # _print('experiment finished, cleaning up...')
  # await run_playbook(
  #     playbook_path=proj_root / 'playbooks/cleanup.yml',
  #     task_name=test_config.task_name,
  #     db_name=test_config.target,
  #     host=[test_config.hosts.tester, test_config.hosts.testee],
  # )

  # reboot...
  # _print('rebooting...')
  # await run_playbook(
  #    playbook_path=proj_root / 'playbooks/reboot.yml',
  #    host=[test_config.hosts.tester, test_config.hosts.testee],
  # )
  # _print('done.')


async def single_test(task_id, rep, clients):
  global deploy_playbook_path
  global tester_playbook_path

  _print(f'{task_id}: carrying out #{rep} repetition test...')
  try:
      # --------------------hadoop------------------
      stdout_hadoop, stderr_hadoop = await run_playbook(
          deploy_hadoop_playbook_path,
          host='wyl1',
          task_id=task_id,
          rep=rep,
      )
      stdout_hadoop, stderr_hadoop = await run_playbook(
          deploy_hadoop_playbook_path,
          host='wyl2',
          task_id=task_id,
          rep=rep,
      )
      stdout_hadoop, stderr_hadoop = await run_playbook(
          deploy_hadoop_playbook_path,
          host='wyl3',
          task_id=task_id,
          rep=rep,
      )
      # - launch test and fetch result
      _print(f'{task_id} - {rep}: hibench testing...')
      # todo---------------------------------------------
      m11, m12 = await run_playbook(
          tester_playbook_path,
          # host="wyl1",
          task_id=task_id,
          rep=rep,
          workload_path=str(workload_path),
          n_client=clients
      )
      # print(m11)
      # print(m12)
      _print(f'{task_id} - {rep}: hibench done.')

      # - cleanup os config

  except RuntimeError as e:
      errlog_path = result_dir / f'{task_id}_error_{rep}.log'
      errlog_path.write_text(str(e))
      print(e)

#todo -------------------------------------------------------------------------------------------------------
#先执行这里
test_config = parse_cmd() #解析出cmd读出yml一个class 里面有众多属性---------------------

# assert test_config is not None

# calculate paths
proj_root = Path(__file__, '../../..').resolve()  # 解析成绝对路径
db_dir = proj_root / f'target/target_spark'
result_dir = db_dir / f'results/spark-rl-test'  #在other里
deploy_spark_playbook_path = db_dir / 'playbook/deploy_spark.yml'
deploy_hadoop_playbook_path = db_dir / 'playbook/deploy_hadoop.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'
workload_path = db_dir / f'workload/work.conf'
app_setting_path = proj_root / 'target/target_spark/workload/allconfig2.yml'
spark_setting_path = proj_root / 'target/target_spark/workload/spark.yml'
yarn_setting_path = proj_root / 'target/target_spark/workload/yarn.yml'
dfs_setting_path = proj_root / 'target/target_spark/workload/dfs.yml'
spark_setting = yaml.load(spark_setting_path.read_text())
yarn_setting = yaml.load(yarn_setting_path.read_text())
dfs_setting = yaml.load(dfs_setting_path.read_text())
app_setting = yaml.load(app_setting_path.read_text())
clients=16
init_id = -1

# check existing results, find minimum available task_id
# exist_task_id = find_exist_task_result()#删除之前的重开或者继续开
# if exist_task_id is not None:
#   _print(f'previous results found, with max task_id={exist_task_id}')
#   policy = test_config.exist
#   if policy == 'delete':
#     for file in sorted(result_dir.glob('*')):
#       file.unlink()
#     _print('all deleted')
#   elif policy == 'continue':
#     _print(f'continue with task_id={exist_task_id + 1}')
#     init_id = exist_task_id
#   else:
#     _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
#     sys.exit(0)

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)

# dump test configs
(result_dir / 'test_config.yml').write_text(
    yaml.dump(test_config, default_flow_style=False)
)
# read parameters for tuning
#event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        init_id=init_id,
        app_setting=app_setting
    )
)
#这个main是调用的  上面的代码会先执行
loop.close()
