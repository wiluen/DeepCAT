## Main codes
in this file, there are main codes of DeepCAT/DeepCAT+ and baseline approaches.

### lib
Call Ansible to configure files, start testing and obtain results.

### logs
In this file, we provide some examples of workload runtime log metrics from Spark event logs, including 18 dimensional metrics.
Spark event logs metrics and the description：
- **job_number**,number of jobs
- **stage_number**,number of stages
- **task_number**,number of tasks
- **task_duration_med**,the median duration of tasks
- **executor_deserialize_time_med**,the median deserialization time of executors
- **executor_deserialize_cputime_med**,the median deserialization cpu time of executors
- **executor_runtime_med**,the median runtime of executors
- **executor_cputime_med**,the median cpu time of executors
- **result_size_med**,the median result sizes
- **gc_time_med**,the median garbage collect time
- **result_serialization_time_med**,the median serialization time of results
- **memory_byte_spilled_med**,the median bytes of memory spill
- **disk_byte_spilled_med**,the median bytes of disk spill
- **blocks_fetched_sum**,total number of data blocks fetched
- **fetchWait_Time_med**,the median waiting time of data to be fetched
- **shuffle_read_sum**,total bytes of data read during shuffle operations
- **totalRecords_Read_sum**,total records from data source
- **shuffle_write_sum**,total bytes of data write during shuffle operations
