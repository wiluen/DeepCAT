import pandas as pd

df = pd.read_csv("/home/wk/sd_hibench/target_spark/spark-bodropout-test/hibench/report/hibench.csv", dtype=object)
data = df.values 
index1 = list(df.keys())  
