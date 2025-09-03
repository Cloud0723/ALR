#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=10
# 创建logs目录
mkdir -p logs

# 遍历不同的uncertainty值
for uncertainty in 0.3 0.5 0.7
do
  # 为每个uncertainty值创建对应的目录
  mkdir -p logs/uncertainty_${uncertainty}
  mkdir -p save_policy/hopper_${uncertainty}
  
  # 对每个uncertainty值运行3个不同的seed
  for seed in 1 2 3
  do
    echo "Running with uncertainty=${uncertainty} and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/uncertainty_${uncertainty}/hopper_seed_${seed}_RL.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # stage 1
      python -u ml/RL.py \
        --config configs/samples/ALR/hopper_RL.yml \
        --output ./save_policy/hopper_${uncertainty}/seed_${seed}/stage1 \
        --uncertainty ${uncertainty} \
      
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "All experiments have been launched. Check logs directory for progress."