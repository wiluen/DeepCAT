# Offline collected datasets of Hibench workloads
This dataset characterizes the performance differences of Spark programs under different configurations.

## Columns
The first 32 columns represent configuration items, and the last column represents execution time.
Note that the parameter values have been normalized.


You can convert it into a real configuration through formula ：**x_true=max((x*conf_max),conf_min)**

## Experimental platform
- local 3-node Spark cluster.
- each vm 8-cores Intel(R) Core(TM) i7-10700 2.9GHz, 16 GB DDR4 memory and 1TB HDD.1Gb/s Ethernet network

## Version of the frameworks 
- Apache Spark 2.2.2
- Hadoop 2.7.3
- Hibench 7.0
- JDK 1.8.0_ 211
