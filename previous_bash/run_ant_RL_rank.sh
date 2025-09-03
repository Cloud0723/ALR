#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=8
# 创建logs目录
mkdir -p logs

# 固定uncertainty值
uncertainty=1

# 要测试的rank值
ranks=(16 24 32 40 48 56 64)

# 为uncertainty值创建对应的目录
mkdir -p logs/ant_ranks

# 遍历不同的rank值
for rank in "${ranks[@]}"
do
  mkdir -p save_policy/ant_rank_${rank}
  
  # 对每个rank值运行不同的seed
  for seed in 1 2 3 4 5
  do
    echo "Running with rank=${rank} and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/ant_ranks/ant_rank_${rank}_seed_${seed}_RL.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # 单次RL训练，使用不同rank的配置
      python -u ml/RL.py \
        --config configs/samples/ALR/ant_RL_${rank}.yaml \
        --output ./save_policy/ant_rank_${rank}/seed_${seed}/model \
        --uncertainty ${uncertainty} \
      
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "所有Ant环境不同rank的RL实验已启动。请在logs目录查看进度。" 