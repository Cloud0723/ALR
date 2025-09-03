#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=4
# 创建logs目录
mkdir -p logs

# 设置固定的uncertainty和beta值
uncertainty=1.0
beta=0.96

# 为uncertainty值创建对应的目录
mkdir -p logs/uncertainty_${uncertainty}
mkdir -p save_policy/humanoid_${uncertainty}_beta_${beta}

# 运行10个不同的seed
for seed in 1 2 3 4 5 6 7 8 9 10
do
  echo "Running with uncertainty=${uncertainty}, beta=${beta} and seed=${seed}"
  
  # 创建日志文件名
  log_file="logs/uncertainty_${uncertainty}/humanoid_seed_${seed}.log"
  
  # 使用nohup运行实验并将输出重定向到日志文件
  nohup bash -c "
    set -x
    set -e
    
    # stage 1
    python -u ml/RL.py \
      --config configs/samples/ALR/humanoid_ALR.yml \
      --output ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage1 \
      --uncertainty ${uncertainty} \
    
    python -u ml/svd_approximation.py \
      --policy_model_path ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage1/HumanoidRandom-v0_epoch_995.pt \
      --output_model_path ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage2/init.pt
    
    for ((i=2; i<${epoch}; i++))
    do
      j=\$((i+1))
      python -u ml/RL.py \
        --config configs/samples/ALR/humanoid_ALR.yml \
        --policy_model_path ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/init.pt \
        --output ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i} \
        --uncertainty ${uncertainty} \
      
      python -u ml/svd_approximation.py \
        --policy_model_path ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/HumanoidRandom-v0_epoch_995.pt \
        --output_model_path ./save_policy/humanoid_${uncertainty}_beta_${beta}/seed_${seed}/stage\${j}/init.pt
    done
  " > ${log_file} 2>&1 &
  
  # 显示进程ID
  echo "Started process with PID $! - Log file: ${log_file}"
  
  # 给每个任务一些启动时间，防止资源竞争
  sleep 2
done

echo "All experiments have been launched. Check logs directory for progress." 