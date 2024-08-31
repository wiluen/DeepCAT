import asyncio
import yaml
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from statistics import mean
from lib.other import parse_cmd, run_playbook, get_default, choose_metric_results
import random
import math
from lib.result_parser import parse_result


# 42
# KNOBS=[
# spark_default_parallelism
# spark_driver_cores
# spark_driver_memory
# spark_executor_memory
# spark_reducer_maxSizeInFlight
# spark_shuffle_compress
# spark_shuffle_file_buffer
# spark_shuffle_spill_compress
# spark_broadcast_compress
# spark_io_compression_codec
# spark_io_compression_blockSize
# spark_rdd_compress
# spark_serializer
# spark_memory_fraction
# spark_memory_storageFraction
# spark_broadcast_blockSize
# spark_executor_cores
# spark_yarn_am_memory
# spark_yarn_am_cores
# spark_executor_instances
# yarn_executor_num
# yarn_executor_core
# ]

proj_root = Path(__file__, '../../..').resolve()

db_dir = proj_root / f'target/target_spark'
result_dir = db_dir / f'results/spark-rl-test'
init_s_dir = db_dir / f'results/initial_state'
setting_path = proj_root / \
               f'target/target_spark/os_configs_info.yml'
deploy_spark_playbook_path = db_dir / 'playbook/deploy_spark.yml'
deploy_hadoop_playbook_path = db_dir / 'playbook/deploy_hadoop.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'
osconfig_playbook_path = db_dir / 'playbook/set_os.yml'
reboot_playbook_path = db_dir / 'playbook/reboot.yml'
workload_path = db_dir / f'workload/work.conf'
app_setting_path = proj_root / 'target/target_spark/workload/allconfig2.yml'

clients=16
def _print(msg):
   print(f'[{datetime.now()}] - {msg}')
    # print('[' + datetime.now() + ']')

result_dir.mkdir(parents=True, exist_ok=True)
app_setting = yaml.load(app_setting_path.read_text())  # pylint: disable=E1101
init_a=[]
#拆分参数--后期有三层
def divide_config(sampled_config,  app_setting):
    for k in sampled_config.keys():
        if type(sampled_config[k]) is bool:
            # make sure no uppercase 'True/False' literal in result
            sampled_config[k] = str(sampled_config[k]).lower()
        elif type(sampled_config[k]) is np.float64:
            sampled_config[k] = float(sampled_config[k])
    sampled_app_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in app_setting)
    )
    return sampled_app_config


def init_conf(i,j):# 可用来看基准duration
    sampled_config=get_default(app_setting)
    print("sampled_config=",sampled_config)
    # sampled_app_config=divide_config(
    #     sampled_config,
    #     app_setting=app_setting
    # )
    app_config_path = result_dir / f'{i}_{j}_app_config.yml'
    app_config_path.write_text(
        yaml.dump(sampled_config, default_flow_style=False)
    )
    _print(f'{i}-{j}: Default:spark_config generated.')


def a2file(action,i,j):
    m=0
    sparkres = {}
    yarnres= {}
    dfsres= {}
    for k, conf in app_setting.items():
        if m<=19:                #13
            if conf['max'] is not None:
                sparkres[k] = conf['max'] * action[m]
                if sparkres[k] >= 1:
                    sparkres[k] = int(sparkres[k])
                    sparkres[k] = max(sparkres[k], conf['min'])
                else:
                    sparkres[k] = format(sparkres[k], '.2f')  # str
                    sparkres[k] = max(float(sparkres[k]), conf['min'])  # error:float and str cannot compare    add some noise
            else:
                #为range形 2,3 ge
                knob_value = conf['range']
                # print(knob_value)
                enum_size = len(knob_value)
                enum_index = int(enum_size * action[m])  # int() 返回整数
                enum_index = min(enum_size - 1, enum_index)
                eval_value = knob_value[enum_index]
                sparkres[k] = eval_value
        elif m<=26: #yarn has float >1           #14#
            yarnres[k] = conf['max'] * action[m]
            # if k == 'yarn_nodemanager_vmem_pmem_ratio':
            #     pass
            # else:
            yarnres[k] = int(yarnres[k])
            yarnres[k] = max(yarnres[k], conf['min'])
        else: # all >1 int             #17
            if conf['max'] is not None:
                dfsres[k] = conf['max'] * action[m]
                dfsres[k] = int(dfsres[k])
                dfsres[k] = max(dfsres[k], conf['min'])
            else:
                knob_value = conf['range']
                # print(knob_value)
                enum_size = len(knob_value)
                enum_index = int(enum_size * action[m])  # int() 返回整数
                enum_index = min(enum_size - 1, enum_index)
                eval_value = knob_value[enum_index]
                dfsres[k] = eval_value
        m+=1
    spark_config_path = result_dir / f'{i}_{j}_spark_config.yml'
    yarn_config_path = result_dir / f'{i}_{j}_yarn_config.yml'
    dfs_config_path = result_dir / f'{i}_{j}_dfs_config.yml'
    spark_config_path.write_text(
        yaml.dump(sparkres, default_flow_style=False)
    )
    yarn_config_path.write_text(
        yaml.dump(yarnres, default_flow_style=False)
    )
    dfs_config_path.write_text(
        yaml.dump(dfsres, default_flow_style=False)
    )
    print(sparkres)
    print(yarnres)
    print(dfsres)
    _print(f'{i}-{j}: configuration file generated...')
   


def test(task_id,rep):
    # 异步调用
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tester(task_id,rep))
    #loop.close()


async def tester(task_id,rep):
    await single_test(
        task_id=task_id,
        rep=rep,
        clients=clients,
    )
# 读的文件是{task_id}_{rep}_app_config.yml    要改tester.yml


async def single_test(task_id, rep, clients):# testee
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
        _print('deploying...')
        _print('testing...')
        # todo---------------------------------------------
        m11,m12=await run_playbook(
            tester_playbook_path,
            # host="wyl1",
            task_id=task_id,
            rep=rep,
            workload_path=str(workload_path),
            n_client=clients
        )
        _print('over...')

       

    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_error_{rep}.log'
        errlog_path.write_text(str(e))
        print(e)


def getduration(i,j):
    duration = parse_result(
        tester_name='hibench',
        result_dir=result_dir,
        task_id=i,
        rep=j,
        printer=_print)
    # print(f'duration={duration}--')
    # 目标是什么  时间
    return duration


def get_state(dir,i,j,loadtype,datasize):
    sysinfo_wyl2=np.loadtxt(dir / f'{i}_wyl2_state_{j}')
    sysinfo_wyl3=np.loadtxt(dir / f'{i}_wyl3_state_{j}')
    # cpu:0-1   io:10-100  mem:1000+m
    x=np.hstack((sysinfo_wyl2,sysinfo_wyl3))
    x=np.array(x)
    load_feature=np.append(loadtype,datasize)
    s=np.append(x,load_feature)
    return s

def Qtune_get_state(i,j):
    # state=6+6
    feature = []
    state=[]
    with open(result_dir / f'{i}_run_result_{j}', 'r') as f:
        seq = f.read()
        time = float(seq.split('\n')[0])
        for i in range(1, 7):
            state_i = float(seq.split('\n')[i])
            feature.append(state_i)
    with open(result_dir / f'{i}_wyl2_state_{j}', 'r') as f2:
        seq2=f2.read()
        seq2=seq2.split(':')
        n1=len(seq2)
        x1=seq2[n1-1].split(',')
        state.append(x1[0])
        state.append(x1[1])
        state.append(x1[2])
    with open(result_dir / f'{i}_wyl3_state_{j}', 'r') as f3:
        seq3=f3.read()
        seq3=seq3.split(':')
        n2=len(seq3)
        x2=seq3[n2-1].split(',')
        state.append(x2[0])
        state.append(x2[1])
        state.append(x2[2])

    return time,feature,state


def getnextstate(i,j,loadtype,datasize):
    dir=result_dir
    s_=get_state(dir,i,j,loadtype,datasize)
    return s_

def init_state(loadtype,datasize):
    dir=init_s_dir
    s=get_state(dir,0,0,loadtype,datasize)
    return s

#old state-->new state
def delstate():
    arr=np.loadtxt('memory_ts/pool.txt', delimiter=' ')
    x = np.delete(arr, [3,4,5,9,10,11,48,49,50,54,55,56], axis=1)
    np.savetxt('memory_ts/pool_newstate.txt', x, fmt='%f', delimiter=' ')


def MLdata():
    # key1,key2,key3,......label
    arr=np.loadtxt('memory_km/pool_real_time.txt', delimiter=' ')
    x = arr[:,8:41]
    np.savetxt('memory_km/dataset.txt', x, fmt='%f', delimiter=' ')


def a2feature():
    Y = np.loadtxt('memory_km/dataset.txt', delimiter=' ')
    # Y=Y[:3]
    content=[]
    for i in range(len(Y)):
        m=0
        sparkres = {}
        yarnres= {}
        dfsres= {}
        action=Y[i]
        for k, conf in app_setting.items():
            if m<=19:                #13
                if conf['max'] is not None:
                    sparkres[k] = conf['max'] * action[m]
                    if sparkres[k] >= 1:
                        sparkres[k] = int(sparkres[k])
                        sparkres[k] = max(sparkres[k], conf['min'])
                    else:
                        sparkres[k] = format(sparkres[k], '.2f')  # str
                        sparkres[k] = max(float(sparkres[k]), conf['min'])  # error:float and str cannot compare    add some noise
                else:
                   
                    knob_value = conf['range']
                    # print(knob_value)
                    enum_size = len(knob_value)
                    enum_index = int(enum_size * action[m])  # int() 返回整数
                    enum_index = min(enum_size - 1, enum_index)
                    eval_value = knob_value[enum_index]
                    sparkres[k] = eval_value
            elif m<=26: #yarn has float >1           #14#
                yarnres[k] = conf['max'] * action[m]
                yarnres[k] = int(yarnres[k])
                yarnres[k] = max(yarnres[k], conf['min'])
            else: # all >1 int             #17
                if conf['max'] is not None:
                    dfsres[k] = conf['max'] * action[m]
                    dfsres[k] = int(dfsres[k])
                    dfsres[k] = max(dfsres[k], conf['min'])
                else:
                    #为range形 2,3 ge
                    knob_value = conf['range']
                    # print(knob_value)
                    enum_size = len(knob_value)
                    enum_index = int(enum_size * action[m])  # int() 返回整数
                    enum_index = min(enum_size - 1, enum_index)
                    eval_value = knob_value[enum_index]
                    dfsres[k] = eval_value
            m+=1
# todo : real configration

        x = []
        y = []
        z = []
        for k1 in sparkres.keys():
            x.append(sparkres[k1])
        for k2 in yarnres.keys():
            y.append(yarnres[k2])
        for k3 in dfsres.keys():
            z.append(dfsres[k3])

        X= np.concatenate((x,y,z))
        # X=np.hstack((x,y))
        if X[4]=='true':
            X[4]=0.5
        else:
            X[4]=1.5

        if X[6]=='false':
            X[6]=0.5
        else:
            X[6]=1.5

        if X[7]=='false':
            X[7]=0.5
        else:
            X[7]=1.5

        if X[8]=='lz4':
            X[8]=0.5
        elif X[8]=='lzf':
            X[8]=1.5
        else:
            X[8]=2.5

        if X[10]=='true':
            X[10]=0.5
        else:
            X[10]=1.5

        if X[11]=='org.apache.spark.serializer.KryoSerializer':
            X[11]=0.5
        else:
            X[11]=1.5

        if X[31] == '64':
            X[31] = 0.5
        elif X[31]=='128':
            X[31] = 1.5
        else:
            X[31]=2.5
# config metric
#         print(X)
        X=X.astype(float)
        content.append(X)
        # print(X)
    np.savetxt('memory_km/datasetforGP.txt', content, fmt='%f',delimiter=' ')
    print('memory pool save..')



def getgood():
    # select good sample from all memory   t<50s
    m=0
    c=np.zeros((1200, 8 * 2 + 32 + 2), dtype=np.float32)
    Y = np.loadtxt('memory_pr/pool_newstate.txt', delimiter=' ')
    for i in range(1225):
        line=Y[i]
        if line[40]<=0.15:
            c[m]=line
            m+=1
    np.savetxt('memory_pr/pool_bad_016.txt', c, fmt='%f', delimiter=' ')
    print('memory pool save..')
    print(f'{m} bad sample')


def deletefail():
    m=0
    c=np.zeros((400, 14 * 2 + 32 + 2), dtype=np.float32)
    Y = np.loadtxt('memory_kmeans/pool_newstate.txt', delimiter=' ')
    for i in range(580):
        line=Y[i]
        if line[46]!=-1:
            c[m]=line
            m+=1
    np.savetxt('memory_kmeans/pool_oo.txt', c, fmt='%f', delimiter=' ')
    print(c)
    print('memory pool save..')



def change_r():
    # select good sample from all memory   t<50s
    m=0
    c=np.zeros((1151, 8 * 2 + 32 + 2), dtype=np.float32)
    Y = np.loadtxt('memory_ts/pool_newstate.txt', delimiter=' ')
    for i in range(1151):
        line=Y[i]
        line[40]=60-60*line[40]
        # if line[46]>0.16:
        c[m]=line
        m+=1
    np.savetxt('memory_ts/pool_realtime.txt', c, fmt='%f', delimiter=' ')
    print('memory pool save..')
    # print(f'{m} good sample')


def _calculate_reward(delta0, deltat):
    if delta0 > 0:
        _reward = ((1 + delta0) ** 2 - 1) * math.fabs(1 + deltat)
    else:
        _reward = - ((1 - delta0) ** 2 - 1) * math.fabs(1 - deltat)

    if _reward > 0 and deltat < 0:
        _reward = 0
    return _reward

def cdbtune_r():
    # note   start0 is my       default is cdbtune
    # not start0 :our expect
    # wc:default40s  pr 80s  km 180s
    reward=[]
    default_dur=180
    line = np.loadtxt('memory_km/pool_real_time.txt', delimiter=' ')
    # -----------------------cdb reward--------------------------
    for i in range(1480):
        dur=line[i][40]
        print(dur)
        if i==0:
            last_dur=180    #dur(t-1)
        else:
            last_dur=line[i-1][40]
        delta_0 = (default_dur - dur) / default_dur  # tendency  >0 good  <0 bad
        delta_t = (last_dur - dur) / last_dur      #vs last      >0 good  <0 bad
        r=_calculate_reward(delta_0,delta_t)
        reward.append(r)
    for i in range(1480):
        line[i][40]=reward[i]

    np.savetxt('memory_km/pool_cdbtune_final.txt', line, fmt='%f', delimiter=' ')









