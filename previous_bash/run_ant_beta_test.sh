#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=4
# 固定uncertainty值
uncertainty=1

# 创建logs目录
mkdir -p logs
mkdir -p logs/ant_beta_test

# 遍历不同的beta值
for beta in 0.94 0.96 0.98
do
  # 为每个beta值创建对应的目录
  mkdir -p logs/ant_beta_test/beta_${beta}
  mkdir -p save_policy/ant_beta_${beta}
  
  # 对每个beta值运行3个不同的seed
  for seed in 1 2 3 4 5 6 7
  do
    echo "Running with uncertainty=${uncertainty}, beta=${beta}, and seed=${seed}"
    
    # 创建日志文件名
    log_file="logs/ant_beta_test/beta_${beta}/ant_seed_${seed}_beta_${beta}.log"
    
    # 使用nohup运行实验并将输出重定向到日志文件
    nohup bash -c "
      set -x
      set -e
      
      # stage 1
      python -u ml/RL.py \
        --config configs/samples/ALR/ant_ALR.yaml \
        --output ./save_policy/ant_beta_${beta}/seed_${seed}/stage1 \
        --uncertainty ${uncertainty} \
      
      python -u ml/svd_approximation.py \
        --policy_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage1/AntRandom-v0_epoch_995.pt \
        --output_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage2/init.pt \
        --beta ${beta}
      
      for ((i=2; i<${epoch}; i++))
      do
        j=\$((i+1))
        python -u ml/RL.py \
          --config configs/samples/ALR/ant_ALR.yaml \
          --policy_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage\${i}/init.pt \
          --output ./save_policy/ant_beta_${beta}/seed_${seed}/stage\${i} \
          --uncertainty ${uncertainty} \
        
        python -u ml/svd_approximation.py \
          --policy_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage\${i}/AntRandom-v0_epoch_995.pt \
          --output_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage\${j}/init.pt \
          --beta ${beta}
      done
      
      # 运行最后一个阶段
      python -u ml/RL.py \
        --config configs/samples/ALR/ant_ALR.yaml \
        --policy_model_path ./save_policy/ant_beta_${beta}/seed_${seed}/stage${epoch}/init.pt \
        --output ./save_policy/ant_beta_${beta}/seed_${seed}/stage${epoch} \
        --uncertainty ${uncertainty} \
      
    " > ${log_file} 2>&1 &
    
    # 显示进程ID
    echo "Started process with PID $! - Log file: ${log_file}"
    
    # 给每个任务一些启动时间，防止资源竞争
    sleep 2
  done
done

echo "All beta tests have been launched. Check logs/ant_beta_test directory for progress." 