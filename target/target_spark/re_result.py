import pandas as pd

df = pd.read_csv("/home/wk/sd_hibench/target_spark/spark-bodropout-test/hibench/report/hibench.csv", dtype=object)
data = df.values  # data是数组，直接从文件读出来的数据格式是数组
index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列�?data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列�?data = pd.DataFrame(data, index=index1)  # 将data的行列转�?data.to_csv('/home/wk/sd_hibench/target_spark/spark-bo-test/hibench/report/' + 'hibench_new.csv', header=0)