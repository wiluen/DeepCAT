# DeepCAT+: A Low-Cost and Transferrable Online Configuration Auto-Tuning Approach for Big Data Frameworks(ICPP22,TPDS revision)
Big data frameworks usually provide a large number of performance-related parameters. Online auto-tuning these parameters based on deep reinforcement learning (DRL) to achieve a better performance has shown their advantages over search-based and machine learning-based approaches. Unfortunately, the time cost during the online tuning phase of conventional DRL-based methods is still heavy, especially for big data applications. To reduce the total online tuning cost and increase the adaptability: 1) DeepCAT+ utilizes the TD3 algorithm instead of DDPG to alleviate value overestimation; 2) DeepCAT+ modifies the conventional experience replay to fully utilize the rare but valuable transitions via a novel reward-driven prioritized experience replay mechanism; 3) DeepCAT+ designs a Twin-Q Optimizer to estimate the execution time of each action without the costly configuration evaluation and optimize the sub-optimal ones to achieve a low-cost exploration-exploitation tradeoff; 4) Furthermore, DeepCAT+ also implements an Online Continual Learner module based on Progressive Neural Networks to transfer knowledge from historical tuning experiences. 
![system overview](https://github.com/wiluen/DeepCAT/blob/main/fig/overview.jpg)
## Start
### Cluster deployment
1. Install Hadoop distributed environment and file system
2. install the Spark computing framework
3. Install and compile the hibench testing framework
4. Install Ansible Playbook for batch configuration and automated deployment

### Steps for reproducing DeepCAT+’s Results
1. Data collection: collect offline exploration data, including cluster metric states, configuration values, rewards.
2. Use the data to form `memory pool` for offline training (`offline_train()` function in `DeepCAT.py`) and save the model.
3. Use the model to tune configuration for big data frameworks using `tune()` in `DeepCAT.py`. Note there are 2 polciy:
   - if the worklaod is **known**, DeepCAT will direct conduct optimization.
   - if the worklaod is **unknown**, DeepCAT will use Progressive Neural Networks for continual learning to enhence it's adaptability.
4. Compare DeepCAT with CDBTune, OtterTune and Qtune baselines
   
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
The data collected based on the local 3-node Spark cluster includes the execution time of 4 spark workloads under different configuration values in the `dataset`. It is then processed into memory pools with `transitions` for reinforcement learning training.

### Configuraiton details
1. Description of the performance-critical parameters From Spark, YARN and HDFS
![Description of the performance-critical parameters From Spark, YARN and HDFS](https://github.com/wiluen/DeepCAT/blob/main/fig/sparkconf.jpg)
2. For experiments on Flink, check [test_kit/ultimate/flink-experimental/readme.md](https://github.com/wiluen/DeepCAT/blob/main/test_kit/ultimate/flink-experimental/readme.md) for more details
