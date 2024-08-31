import asyncio
import json
import os
import random
import numpy as np
import shutil
from pathlib import Path
import sys
import yaml
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils import parse_cmd, trans_numeric_to_config

result_dir = Path("/home/zyx/workspace/testFlink/results")


async def run_playbook(playbook_path, tags='all', **extra_vars):
    vars_json_str = json.dumps(extra_vars)
    command = f'ansible-playbook {playbook_path} --extra-vars=\'{vars_json_str}\' --tags={tags}'
    print(command)
    process = await asyncio.create_subprocess_shell(
        cmd=command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f'Error running playbook. Below is stdout:\n {stdout.decode("utf-8")}\nand stderr: {stderr.decode("utf-8")}\n')
    return stdout.decode('utf-8'), stderr.decode('utf-8')


async def single_test(master, workload, task_name, task_id, rep,
                      recordsPerInterval, objective, test_time, exception_num=3):
    tester_playbook_path = Path("/home/zyx/workspace/testFlink/src/tester.yml")
    vm_list_info_path = Path("/home/zyx/workspace/testFlink/conf/vm_lists.yml")
    vm_list = yaml.load(vm_list_info_path.read_text(), Loader=yaml.FullLoader)
    result_dir = Path("/home/zyx/workspace/testFlink/results")/task_name

    try:
        if exception_num <= 0:
            return

        print(f'{task_id} - {rep}: testing...')
        stdout, stderr = await run_playbook(
            tester_playbook_path,
            master=master,
            workload=workload,
            task_name=task_name,
            task_id=task_id,
            task_rep=rep,
            recordsPerInterval=recordsPerInterval,
            objective=objective,
            test_time=test_time,
            VMs=vm_list
        )

        out_log_path = result_dir / f'{task_id}_test_out.log'
        out_log_path.write_text(stdout)
        print(f'{task_id} - {rep}: test done.')

    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_test_error_{rep}.log'
        errlog_path.write_text(str(e))
        print(e)
        await single_test(master, workload, task_name, task_id,
                          rep, recordsPerInterval, objective, 1, exception_num=exception_num - 1)


async def main(test_config):
    rep = 0
    recordsPerInterval = 1000
    task_result_path = Path("../results")/test_config.task_name
    for task_id in range(test_config.sample_num + 1):
        print(f'第{task_id}轮采样')
        if task_id == 0:  
            default_conf_path = Path("../flink_configs/0_app_config.yml")
            
            shutil.copy2(default_conf_path, task_result_path/"0_app_config.yml")
            await single_test(
                master=test_config.master.to_json(),
                workload=test_config.workload,
                task_name=test_config.task_name,
                task_id=task_id,
                rep=rep,
                recordsPerInterval=recordsPerInterval,
                objective=test_config.streaming_system,
                test_time=test_config.test_time
                # test_time = test_config.test_time
                # task_time=int(test_config.task_time)
            )
        else:
            # find_random_conf(task_id, task_result_path)
            random_collect(task_id, task_result_path)
            await single_test(
                master=test_config.master.to_json(),
                workload=test_config.workload,
                task_name=test_config.task_name,
                task_id=task_id,
                rep=rep,
                recordsPerInterval=recordsPerInterval,
                objective=test_config.streaming_system,
                test_time=test_config.test_time
                # test_time = test_config.test_time
                # task_time=int(test_config.task_time)
            )
        # 获取数据，存入文件


def find_random_conf(task_id, task_result_path):
    proj_root = Path(__file__, '../..').resolve()
    flink_config_setting_path = proj_root / 'flink_configs/flink_conf.yml'
    flink_configs_setting = yaml.load(flink_config_setting_path.read_text(), Loader=yaml.FullLoader)

    numeric_flink_config = {}
    for config_name, config_settings in flink_configs_setting.items():
        has_range = config_settings.get('range')
        if has_range:
            numeric_flink_config[config_name] = random.randint(0, len(config_settings['range']) - 1)
        else:
            min_value = config_settings['min']
            max_value = config_settings['max']
            isFloat = bool(config_settings['float'])
            if not isFloat:
                numeric_flink_config[config_name] = random.randint(min_value, max_value)
            else:
                numeric_flink_config[config_name] = random.uniform(min_value, max_value)
    sample_config = trans_numeric_to_config(numeric_flink_config)
    print(sample_config)
    #写进app config yml文件
    with open(task_result_path / f'{task_id}_app_config.yml', 'w') as file:
        yaml.dump(sample_config, file)

def random_collect(task_id, task_result_path):
    random_data=[]
    # x*(max-min)+min
    proj_root = Path(__file__, '../..').resolve()
    flink_config_setting_path = proj_root / 'flink_configs/flink_conf.yml'
    flink_configs_setting = yaml.load(flink_config_setting_path.read_text(), Loader=yaml.FullLoader)

    numeric_flink_config = {}
    for config_name, config_settings in flink_configs_setting.items():
        has_range = config_settings.get('range')
        if has_range:
            # numeric_flink_config[config_name] = random.randint(0, len(config_settings['range']) - 1)
            x=random.random()
            if x>0.5:
                numeric_flink_config[config_name]=1
            else:
                numeric_flink_config[config_name]=0
        else:
            min_value = config_settings['min']
            max_value = config_settings['max']
            # isFloat = bool(config_settings['float'])
            # if not isFloat:
            #     numeric_flink_config[config_name] = random.randint(min_value, max_value)
            # else:
            x=random.random()
            numeric_flink_config[config_name] = int(x*(max_value-min_value)+min_value)
        random_data.append(x)
    sample_config = trans_numeric_to_config(numeric_flink_config)
    print(random_data)
    random_data=np.array(random_data)
    np.savetxt(f'/home/zyx/workspace/testFlink/results/wc1k_collect/{task_id}_action.txt', random_data, delimiter=',', fmt='%.2f')
    print(sample_config)
    #写进app config yml文件
    with open(task_result_path / f'{task_id}_app_config.yml', 'w') as file:
        yaml.dump(sample_config, file)

test_config = parse_cmd()
os.makedirs(f"../results/{test_config.task_name}", exist_ok=True)
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        # app_setting=app_setting,
    )
)
loop.close()
