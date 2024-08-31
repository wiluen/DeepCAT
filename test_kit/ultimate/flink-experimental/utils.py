import re
import traceback
from datetime import datetime
from pathlib import Path
from math import inf
import yaml


def trans_numeric_to_config(numeric):
    config = {}
    proj_root = Path(__file__, '../..').resolve()
    flink_config_setting_path = proj_root / 'flink_configs/flink_conf.yml'
    flink_configs_setting = yaml.load(flink_config_setting_path.read_text(), Loader=yaml.FullLoader)
    for config_name, config_settings in flink_configs_setting.items():
        has_range = config_settings.get('range')
        num = numeric[config_name]
        if has_range:
            config[config_name] = flink_configs_setting[config_name]['range'][num]
        else:
            config[config_name] = num
    return config


def assign(obj, path, value):
    keys = path.split('.')
    for k in keys[:-1]:
        v = obj.get(k)
        if v is None:
            obj[k] = {}
        elif type(v) is not dict:
            raise Exception(f'error while assigning {path} with {value} on {obj}.')
        obj = obj[k]
    try:
        value = int(value)
    except:
        pass
    if str(value) in ('yes', 'true'):
        value = True
    if str(value) in ('no', 'false'):
        value = False
    obj[keys[-1]] = value


class Master:
    def __init__(self, obj):
        self.name = obj['name']
        self.ip = obj['ip']
        self.user = obj['user']
        self.password = obj['password']

    def to_json(self):
        return {
            'name': self.name,
            'ip': self.ip,
            'user': self.user,
            'password': self.password
        }


class OptimizerConfig:
    def __init__(self, obj):
        self.name = obj['name']
        self.iter_limit = obj['iter_limit']
        self.repitition = obj['repitition']
        self.recordsPerInterval = obj['recordsPerInterval']
        self.stepSize = obj['stepSize']
        self.tune_cloud = obj['tune_cloud']
        self.tune_app = obj['tune_app']
        self.extra_vars = obj.get('extra_vars', {})

        if self.iter_limit < 0:
            self.iter_limit = inf


class TestConfig:
    def __init__(self, obj):
        self.task_name = obj['task_name']
        self.streaming_system = obj['streaming_system']
        self.workload = obj['workload']
        self.master = Master(obj['master'])
        self.test_time = int(obj['test_time'])
        self.sample_num = int(obj['sample_num'])


def parse_cmd():
    try:
        # MEIassert len(args) > 1, 'too few arguments'
        # conf_path = ../../target/hbase/tests/bo-ei.y ml
        # *others = task_name=janusgraph_hbase_bo_1000 exist=delete
        others = []

        # 读出bo-ei.yml
        conf = yaml.load(
            Path("conf.yml").resolve().read_text(),  # pylint: disable=E1101
            # "/home/zyx/workspace/testFlink/src/conf.yml",
            Loader=yaml.FullLoader
        )
        regexp = re.compile(r'(.+)=(.+)')
        for other in others:
            match = regexp.match(other.strip())
            k, v = match.groups()
            assign(conf, k, v)
        print(conf)
        return TestConfig(conf)
    except Exception as e:
        print(e.args)
        print(traceback.print_stack(e))
        print('Usage: python run.py <path_to_conf> [path.to.attr=value]')
