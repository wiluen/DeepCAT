## Main codes
in this file, there are main codes of DeepCAT/DeepCAT+ and baseline approaches.
- DeepCAT.py: training and online tuning of DeepCAT
- DeepCAT_with_PNN.py: online tuning of DeepCAT+ to use PNN for unknown workloads
  1. choose most similar historical workload based on running logs
  2. load the base model and initial a new policy model
  3. update the policy model by PNN and conduct online tuning
- DDPG.py and TD3.py: original reinforcement learning algorithms
- SparkENV.py: the `environment` module of RL in BD configuration tuning
- util.py: some important implementations in SparkENV.py

### lib
Call `Ansible` to configure files, start testing and obtain results.

### logs
In this file, we provide some examples of workload runtime log metrics from Spark event logs, including 18 dimensional metrics.
Spark event logs metrics and the description：
- **job_number**, number of jobs
- **stage_number**, number of stages
- **task_number**, number of tasks
- **task_duration_med**, the median duration of tasks
- **executor_deserialize_time_med**, the median deserialization time of executors
- **executor_deserialize_cputime_med**, the median deserialization cpu time of executors
- **executor_runtime_med**, the median runtime of executors
- **executor_cputime_med**, the median cpu time of executors
- **result_size_med**, the median result sizes
- **gc_time_med**, the median garbage collect time
- **result_serialization_time_med**, the median serialization time of results
- **memory_byte_spilled_med**, the median bytes of memory spill
- **disk_byte_spilled_med**, the median bytes of disk spill
- **blocks_fetched_sum**, total number of data blocks fetched
- **fetchWait_Time_med**, the median waiting time of data to be fetched
- **shuffle_read_sum**, total bytes of data read during shuffle operations
- **totalRecords_Read_sum**, total records from data source
- **shuffle_write_sum**, total bytes of data write during shuffle operations

`workload_similar.py` provide an example to calculate the similarity between different workloads, and the DeepCAT+ select the existing model trained with the most similar historcial workload as the base model, and utilize PNN for optimization.

### memory
memory pools of RL training. consisting of (state,action,reward,next_state) each rows.

### model

### flink-experimental
For the codes and experiments on Flink version, check [test_kit/ultimate/flink-experimental/readme.md](https://github.com/wiluen/DeepCAT/blob/main/test_kit/ultimate/flink-experimental/readme.md) for more details.
