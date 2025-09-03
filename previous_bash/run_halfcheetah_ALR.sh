#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=8
# 创建logs目录
mkdir -p logs

# 遍历不同的uncertainty值
for uncertainty in 0.3 0.5 0.7
do
  # 为每个uncertainty值创建对应的目录
  mkdir -p logs/uncertainty_${uncertainty}
  mkdir -p save_policy/halfcheetah_${uncertainty}
  
  # 对每个uncertainty值运行3个不同的seed
  for seed in 1 2 3
  do
    echo "Running with uncertainty=${uncertainty} and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/uncertainty_${uncertainty}/halfcheetah_seed_${seed}.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # stage 1
      python -u ml/RL.py \
        --config configs/samples/ALR/halfcheetah_ALR.yml \
        --output ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage1 \
        --uncertainty ${uncertainty} \
      
      python -u ml/svd_approximation.py \
        --policy_model_path ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage1/HalfCheetahRandom-v0_epoch_495.pt \
        --output_model_path ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage2/init.pt
      
      for ((i=2; i<${epoch}; i++))
      do
        j=\$((i+1))
        python -u ml/RL.py \
          --config configs/samples/ALR/halfcheetah_ALR.yml \
          --policy_model_path ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage\${i}/init.pt \
          --output ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage\${i} \
          --uncertainty ${uncertainty} \
        
        python -u ml/svd_approximation.py \
          --policy_model_path ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage\${i}/HalfCheetahRandom-v0_epoch_495.pt \
          --output_model_path ./save_policy/halfcheetah_${uncertainty}/seed_${seed}/stage\${j}/init.pt
      done
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "All experiments have been launched. Check logs directory for progress." 