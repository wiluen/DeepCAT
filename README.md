# DeepCAT+: A Low-Cost and Transferrable Online Configuration Auto-Tuning Approach for Big Data Frameworks([ICPP22](https://dl.acm.org/doi/abs/10.1145/3545008.3545018),[TPDS24](https://doi.org/10.1109/TPDS.2024.3459889))
Big data frameworks usually provide a large number of performance-related parameters. Online auto-tuning these parameters based on deep reinforcement learning (DRL) to achieve a better performance has shown their advantages over search-based and machine learning-based approaches. Unfortunately, the time cost during the online tuning phase of conventional DRL-based methods is still heavy, especially for big data applications. To reduce the total online tuning cost and increase the adaptability: 1) DeepCAT+ utilizes the TD3 algorithm instead of DDPG to alleviate value overestimation; 2) DeepCAT+ modifies the conventional experience replay to fully utilize the rare but valuable transitions via a novel **reward-driven prioritized experience replay mechanism**; 3) DeepCAT+ designs a **Twin-Q Optimizer** to estimate the execution time of each action without the costly configuration evaluation and optimize the sub-optimal ones to achieve a low-cost exploration-exploitation tradeoff; 4) Furthermore, DeepCAT+ also implements an **Online Continual Learner module** based on Progressive Neural Networks to transfer knowledge from historical tuning experiences. 
![system overview](https://github.com/wiluen/DeepCAT/blob/main/fig/overview.jpg)

## New features in DeepCAT+ beyond [ICPP22 Paper DeepCAT](https://dl.acm.org/doi/abs/10.1145/3545008.3545018)
Progressive Neural Networks (PNN) based Online Continual Learner to enhance the adaptability for dynamic workloads and hardware environments changes.
1. **Log-based workload features extraction** 
2. **PNN-based knowledge transfer**
   
<img src="https://github.com/wiluen/DeepCAT/blob/main/fig/pnn.jpg" alt="pnn" width="500"/>


## Start
### Cluster deployment
1. Install Hadoop distributed environment and file system.
2. install the Spark computing framework.
3. Install and compile the hibench testing framework.
4. Install Ansible Playbook for batch configuration and automated deployment.

### Steps for reproducing DeepCAT+’s Results
1. Data collection: collect offline exploration data, including cluster metric states, configuration values, rewards. The interaction between Python programs and clusters is conducted through `Ansible` tools, check [target/target_spark/readme.md](https://github.com/wiluen/DeepCAT/blob/main/target/target_spark/readme.md) for more details.
2. Use the data to form `memory pool` for offline training and save the model, see `offline_train()` function in `DeepCAT.py`.
3. Use the model to tune configuration for big data frameworks using `tune()` in `DeepCAT.py`. Note there are two polcies:
   - if the workload is **known**, DeepCAT+ will direct conduct optimization, details in `DeepCAT.py`.
   - if the workload is **unknown**, DeepCAT+ will use Progressive Neural Networks for continual learning to enhence it's adaptability, details in `DeepCAT_with_PNN.py`.
4. Compare DeepCAT with CDBTune, OtterTune and Qtune baselines.
   
### Environment Version
- Hadoop 2.7.3
- Spark 2.2.2
- Hibench 7.0
- Ansible

### Install dependencies (with python 3.8)
> pip install -r requirements.txt

### Benchmark
we use 9 worklaods with different input data sizes form Hibench [The HiBench Benchmark Suite: Characterization of the MapReduce-Based Data Analysis](https://www.spec.org/sources/cloudiaas2018/sources/hibench/HiBench/WISS10_conf_full_011.pdf)
- WordCount (WC)
- TeraSort (TS)
- PageRank (PR)
- KMeans (KM)
- Gradient Boosted Trees(GBT)
- Nweight (NW)
- Principal Component Analysis (PCA)
- Aggregation (AGG)
- WordCount(for streaming)

### Baseline
- [Ottertune](https://dl.acm.org/doi/abs/10.1145/3035918.3064029)
- [CDBTune](https://dl.acm.org/doi/abs/10.1145/3299869.3300085)
- [Qtune](https://dl.acm.org/doi/abs/10.14778/3352063.3352129)

### Datasets
1. The data collected based on the local 3-node Spark cluster includes the execution time of 4 spark workloads under different configuration values in the `dataset`, check [dataset](https://github.com/wiluen/DeepCAT/tree/main/dataset) for more details.
2. For reinforcement learning training, memory pools consist of `transitions`(s,a,r,s') is in [test_kit/ultimate/memory](https://github.com/wiluen/DeepCAT/tree/main/test_kit/ultimate/memory) 

### Configuraiton details
1. Description of the performance-critical parameters From Spark, YARN and HDFS
![Description of the performance-critical parameters From Spark, YARN and HDFS](https://github.com/wiluen/DeepCAT/blob/main/fig/sparkconf.jpg)
2. For experiments on Flink, check [test_kit/ultimate/flink-experimental/readme.md](https://github.com/wiluen/DeepCAT/blob/main/test_kit/ultimate/flink-experimental/readme.md) for more details.

### Hardware environments
Seven experimental clusters to extensively evaluate the effectiveness of DeepCAT/DeepCAT+ and its robustness to various hardware environments.
|  Cluster   | Nodes |Cluster types|BD frameworks|Evaluation|
|  :----:  | :----:  | :----:  | :----:  | :----:  |
| Cluster_A  | 3 | Physical machines | Spark | Effectiveness |
| Cluster_B | 3 | VMs_1 | Spark | Adaptability |
| Cluster_C  | 6 | Physical machines | Flink | Other BD frameworks|
| Cluster_D  | 5 | Physical machines + VMs_2 | Spark |Heterogeneous clusters|
| Cluster_E | 8 | VMs_3 | Spark |Large-scale clusters|
| Cluster_F  | 10 | VMs_3 | Spark |Large-scale clusters|
| Cluster_G  | 12 | VMs_3 | Spark |Large-scale clusters|

- Physical machines: 8 cores, 16GB memory
- VMs_1: 8 cores, 8GB memory
- VMs_2: 12 cores, 8GB memory
- VMs_3: 8 cores, 16GB memory
