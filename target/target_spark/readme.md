This folder contains files for interacting with the test cluster.
## role of playbook
- fill configuration files
- replace target configuration files of HADOOP,SPARK,HIBENCH
- start testing
- aggregate test results

## other: Files that need to be replaced for the configuration to take effect.
- Hadoop/etc/hadoop/hdfs-site.xml
- Hadoop/etc/hadoop/yarn-site.xml
- hibench/conf/spark/spark.conf
- hibench/conf/spark/hibench.conf

## sparkconf.yml: contains 32 performance-critical configuraiton parameters.
