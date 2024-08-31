---
- hosts: "{{master.name}}"
  remote_user: "{{master.user}}"
  gather_facts: false

  vars:
    # required extra vars:
    #   - host
    #   - task_name
    #   - task_id
    #   - task_rep
    ansible_sudo_pass: "{{master.password}}"
    db_name: Clusters
    apt_requirements:
      # - libcurl4
    # deploy_home: "/home/{{master.user}}/Desktop/code/{{db_name}}/HiBench"
    Hibench_home: "/home/zyx/Program/HiBench"
    deploy_flink_home: "/home/zyx/Program/flink"
    deploy_spark_home: "/home/{{master.user}}/Desktop/code/{{db_name}}/spark/spark-2.2.2-bin-hadoop2.7"
    deploy_hadoop_home:  "/home/{{master.user}}/Desktop/code/{{db_name}}/hadoop/hadoop-2.7.7"
    deploy_kafka_home: "/home/zyx/Program/kafka"
    local_benchmark_src: ../HiBench.zip
    # db_server: "{{Hibench_home}}/HiBench"
    db_server: "{{Hibench_home}}"
    db_config: "{{db_server}}/conf.yml/"
    db_port: 9092
    flink_web_port: 8081
    spark_port: 8080
    run_results_bin: "{{db_server}}/results"
    collect_results: "{{db_server}}/report"
    local_result_dir: "../results/{{task_name}}"
    local_flink_conf: "../flink_configs/flink-conf.yaml"
    flink_config: "{{deploy_flink_home}}/conf/"
    local_kafka_conf: "../kafka_configs/server.properties"
    kafka_config: "{{deploy_kafka_home}}/config/"

  pre_tasks: # set up a clean env
    - name: create folders
      with_items:
        - "{{run_results_bin}}"
        - "{{collect_results}}"
      file:
        path: "{{item}}"
        state: directory
        recurse: yes
#    - name: stop All
#      shell: "/home/zyx/workspace/shell/stopAll.sh"

    - name: stop flink
      shell: "{{deploy_flink_home}}/bin/stop-cluster.sh"
      when: objective == "flink"
    - name: stop kafka
      shell: "{{deploy_flink_home}}/bin/stop-cluster.sh"
      when: objective == "flink"
  tasks:
    - name: load app_config information
      include_vars:
        file: "{{local_result_dir}}/{{task_id}}_app_config.yml"
        name: app_config
    - name: copy {{objective}} conf
      template:
        src: "{{local_flink_conf}}"
        dest: "{{flink_config}}"
      when: objective == "flink"
#    - name: copy kafka_server.properties
#      template:
#        src: "{{local_kafka_conf}}"
#        dest: "{{kafka_config}}"

#    - name: distribute kafka config
#      shell: "scp -r {{kafka_config}}server.properties {{item.1.user}}@{{item.1.ip}}:/home/zyx/Program/kafka/config/"
#      when: item.0 != 0 and objective == "flink"
#      with_indexed_items:
#        - "{{VMs}}"
#    - name: distinguish broker id
#      shell: echo "broker.id={{item.0 + 1}}" >> {{kafka_config}}server.properties
#      delegate_to: "{{item.1.name}}"
#      with_indexed_items:
#        - "{{VMs}}"
    - name: distribute flink config
      shell: "scp -r {{flink_config}}/flink-conf.yaml {{item.1.user}}@{{item.1.ip}}:/home/zyx/Program/flink/conf/"
      when: item.0 != 0 and objective == "flink"
      with_indexed_items:
        - "{{VMs}}"

    - name: clear report
      file:
        path: "{{db_server}}/report"
        state: "{{item}}"
      with_items:
        - absent
        - directory
    - name: start flink
      shell: "nohup {{deploy_flink_home}}/bin/start-cluster.sh"
      when: objective == "flink"
    - name: wait...
      wait_for:
        host: "{{master.ip}}"
        port: "{{flink_web_port}}"
        delay: 5 # wait 3 seconds till it initialized?
      when: objective == "flink"
#    - name: start All
#      shell: "/home/zyx/workspace/shell/startAll.sh"
#    - name: sending wait ....
#      shell: sleep 20

    - name: generate data
      shell: "nohup {{Hibench_home}}/bin/workloads/streaming/{{workload}}/prepare/dataGen.sh -s 1>{{run_results_bin}}/runResult1 2>{{run_results_bin}}/runError1"
      async: 300
      poll: 0
    - name: sending wait ....
      shell: sleep 5
    - name: consume data
      shell: "nohup {{db_server}}/bin/workloads/streaming/{{workload}}/{{objective}}/run.sh -s 1>{{run_results_bin}}/runResult 2>{{run_results_bin}}/runError"
      async: 300
      poll: 0
    - name: testing wait ....
      shell: sleep 310
      async : 310
      poll: 2

    - name: fetch send data result
      fetch:
        src: "{{run_results_bin}}/runResult1"
        dest: "{{local_result_dir}}/{{task_id}}_send_data_{{task_rep}}_result"
        flat: yes
    - name: fetch send data error
      fetch:
        src: "{{run_results_bin}}/runError1"
        dest: "{{local_result_dir}}/{{task_id}}_send_data_{{task_rep}}_error"
        flat: yes
    - name: fetch consume data result
      fetch:
        src: "{{run_results_bin}}/runResult"
        dest: "{{local_result_dir}}/{{task_id}}_consume_data_{{task_rep}}_result"
        flat: yes
    - name: fetch consume data error
      fetch:
        src: "{{run_results_bin}}/runError"
        dest: "{{local_result_dir}}/{{task_id}}_consume_data_{{task_rep}}_error"
        flat: yes

    - name: stop flink
      shell: "nohup {{deploy_flink_home}}/bin/stop-cluster.sh"
      when: objective == "flink"
    - name: collect metrics
      shell: kafka_topic="$(grep -o '{{objective |upper}}_{{workload}}[0-9_]\+' {{run_results_bin}}/runResult | tail -1)";{{db_server}}/bin/workloads/streaming/{{workload}}/common/metrics_reader_cmd.sh $kafka_topic -s 1>{{run_results_bin}}/collectResult 2>{{run_results_bin}}/collectError;mv {{db_server}}/report/$kafka_topic.csv {{db_server}}/report/test.csv;
      #shell: kafka_topic="$(grep -o '{{objective |upper}}_{{workload}}[0-9_]\+' {{run_results_bin}}/runResult | tail -1)"; JAVA_HOME=/usr/lib/jvm/jdk1.8.0_211 {{db_server}}/bin/workloads/streaming/{{workload}}/common/metrics_reader.sh -s 1  > {{run_results_bin}}/collectResult 2> {{run_results_bin}}/collectError && mv -f {{db_server}}/report/$kafka_topic.csv {{db_server}}/report/test.csv;
      args:
        executable: /bin/bash
      register: result
      until: result.rc == 0
      retries: 5
      delay: 10

    - name: fetch collect data result
      fetch:
        src: "{{run_results_bin}}/collectResult"
        dest: "{{local_result_dir}}/{{task_id}}_collect_{{task_rep}}_result"
        flat: yes
    - name: fetch collect data error
      fetch:
        src: "{{run_results_bin}}/collectError"
        dest: "{{local_result_dir}}/{{task_id}}_collect_{{task_rep}}_error"
        flat: yes
    - name: fetch run result
      fetch:
        src: "{{db_server}}/report/test.csv"
        dest: "{{local_result_dir}}/{{task_id}}_run_result_{{task_rep}}.csv"
        flat: yes
