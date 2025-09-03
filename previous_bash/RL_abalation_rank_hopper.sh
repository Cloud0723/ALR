#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=10

# 创建logs和Figures目录
mkdir -p logs
mkdir -p Figures

# 遍历不同的uncertainty值
for uncertainty in 0.3 0.5 0.7
do
  # 为每个uncertainty值创建对应的目录
  mkdir -p logs/uncertainty_${uncertainty}
  
  # 遍历不同的rank值
  for rank in 16 24 32 40 48
  do
    # 为每个uncertainty和rank组合创建对应的目录
    mkdir -p save_policy/hopper_${uncertainty}_rank_${rank}
    
    # 对每个组合运行3个不同的seed
    for seed in 1 2 3
    do
      echo "Running with uncertainty=${uncertainty}, rank=${rank}, seed=${seed}"
      
      # 创建日志文件名
      log_file="logs/uncertainty_${uncertainty}/hopper_seed_${seed}_rank_${rank}_RL.log"
      
      # 使用nohup运行实验并将输出重定向到日志文件
      nohup bash -c "
        set -x
        set -e
        
        # 运行实验
        python -u ml/RL.py \
          --config configs/samples/ALR/hopper_RL_${rank}.yml \
          --output ./save_policy/hopper_${uncertainty}_rank_${rank}/seed_${seed}/model \
          --uncertainty ${uncertainty}
        
      " > ${log_file} 2>&1 &
      
      # 显示进程ID
      echo "Started process with PID $! - Log file: ${log_file}"
      
      # 给每个任务一些启动时间，防止资源竞争
      sleep 5
    done
  done
done

echo "All experiments have been launched. Check logs directory for progress."
echo "Total experiment combinations: 3 uncertainties × 6 ranks × 3 seeds = 54 runs" 