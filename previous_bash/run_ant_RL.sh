#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=8
# 创建logs目录
mkdir -p logs

# 遍历不同的uncertainty值
for uncertainty in 0.3 0.5 0.7
do
  # 为每个uncertainty值创建对应的目录
  mkdir -p logs/ant_uncertainty_${uncertainty}
  mkdir -p save_policy/ant_${uncertainty}
  
  # 对每个uncertainty值运行不同的seed
  for seed in 1 2 3 4 5 6 7
  do
    echo "Running with uncertainty=${uncertainty} and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/ant_uncertainty_${uncertainty}/ant_seed_${seed}_RL.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # 单次RL训练，无需ALR迭代
      python -u ml/RL.py \
        --config configs/samples/ALR/ant_RL.yml \
        --output ./save_policy/ant_${uncertainty}/seed_${seed}/stage1 \
        --uncertainty ${uncertainty} \
      
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "所有Ant环境RL实验已启动。请在logs目录查看进度。" 