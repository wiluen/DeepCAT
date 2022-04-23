import os
line_all = []
str_key = []
str_value = []
if os.path.exists('/home/wk/sd_hibench/spark/spark-rl-test/hibench/report/hibench.report'):
    with open('/home/wk/sd_hibench/spark/spark-rl-test/hibench/report/hibench.report') as f:
        c=0
        for line in f:
            c+=1
    #        print(type(line), '\n')
            line_all.append(line)
    str_key = line_all[0].split()
    if c==4:   #head + 3 line result       #4
        str_value1 = line_all[1].split()
        str_value2 = line_all[2].split()
        str_value3 = line_all[3].split()
        with open('/home/wk/sd_hibench/spark/spark-rl-test/hibench/report/hibench', 'a+') as f2:
            for i in range(len(str_key)):
                f2.writelines(str_key[i]+', '+str_value1[i]+os.linesep)
            for i in range(len(str_key)):
                f2.writelines(str_key[i]+', '+str_value2[i]+os.linesep)
            for i in range(len(str_key)):
                f2.writelines(str_key[i]+', '+str_value3[i]+os.linesep)
    else:
        with open('/home/wk/sd_hibench/spark/spark-rl-test/hibench/report/hibench', 'a+') as f2:
            f2.writelines("Duration(s), 240")

else:
    with open('/home/wk/sd_hibench/spark/spark-rl-test/hibench/report/hibench', 'a+') as f2:
            f2.writelines("Duration(s), 240")


