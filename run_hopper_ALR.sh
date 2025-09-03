#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=8
# 创建logs目录
mkdir -p logs

# 遍历不同的uncertainty值
for uncertainty in 0.3 0.5
do
  mkdir -p logs/uncertainty_${uncertainty}
  
  # 固定beta值为0.96
  beta=0.96
  
  # 为每个uncertainty和beta值创建对应的目录
  mkdir -p save_policy/hopper_${uncertainty}_beta_${beta}
  
  # 对每个配置运行5个不同的seed
  for seed in 1 2 3 4 5 6 7
  do
    echo "Running with uncertainty=${uncertainty}, beta=${beta} and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/uncertainty_${uncertainty}/hopper_seed_${seed}.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # stage 1
      python -u ml/RL.py \
        --config configs/samples/ALR/hopper_ALR.yml \
        --output ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage1 \
        --uncertainty ${uncertainty}
      
      python -u ml/svd_approximation.py \
        --policy_model_path ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage1/HopperRandom-v0_epoch_995.pt \
        --output_model_path ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage2/init.pt
      
      for ((i=2; i<${epoch}; i++))
      do
        j=\$((i+1))
        python -u ml/RL.py \
          --config configs/samples/ALR/hopper_ALR.yml \
          --policy_model_path ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/init.pt \
          --output ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i} \
          --uncertainty ${uncertainty} 
        
        python -u ml/svd_approximation.py \
          --policy_model_path ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/HopperRandom-v0_epoch_995.pt \
          --output_model_path ./save_policy/hopper_${uncertainty}_beta_${beta}/seed_${seed}/stage\${j}/init.pt
      done
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "All experiments have been launched. Check logs directory for progress." 