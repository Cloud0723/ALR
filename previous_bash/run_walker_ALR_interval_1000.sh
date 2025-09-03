#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=3
# 创建logs目录
mkdir -p logs

# 要测试的参数
uncertainties=(0.5)
betas=(0.96)
seeds=(1 2 3 4 5)  # 修正数组格式
interval=1000

# 遍历不同的uncertainty和beta值
for uncertainty in "${uncertainties[@]}"
do
  # 为每个uncertainty值创建对应的目录
  mkdir -p logs/interval_${interval}_uncertainty_${uncertainty}
  
  for beta in "${betas[@]}"
  do
    # 为每个beta值创建对应的目录
    mkdir -p save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}
    
    # 对每个组合运行不同的seed
    for seed in "${seeds[@]}"
    do
      echo "Running with interval=${interval}, uncertainty=${uncertainty}, beta=${beta}, seed=${seed}"
      
      # 创建日志文件名
      log_file="logs/interval_${interval}_uncertainty_${uncertainty}/walker_seed_${seed}.log"
      
      # 使用nohup运行实验并将输出重定向到日志文件
      nohup bash -c "
        set -x
        set -e
        
        # stage 1
        python -u ml/RL.py \
          --config configs/samples/ALR/walker2d_ALR_interval_1000.yml \
          --output ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage1 \
          --uncertainty ${uncertainty}
        
        python -u ml/svd_approximation.py \
          --policy_model_path ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage1/Walker2dRandom-v0_epoch_995.pt \
          --output_model_path ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage2/init.pt \
          --beta ${beta}
        
        for ((i=2; i<${epoch}; i++))
        do
          j=\$((i+1))
          python -u ml/RL.py \
            --config configs/samples/ALR/walker2d_ALR_interval_1000.yml \
            --policy_model_path ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/init.pt \
            --output ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i} \
            --uncertainty ${uncertainty}
          
          python -u ml/svd_approximation.py \
            --policy_model_path ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/Walker2dRandom-v0_epoch_995.pt \
            --output_model_path ./save_policy/walker_interval_${interval}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${j}/init.pt \
            --beta ${beta}
        done
      " > ${log_file} 2>&1 &
      
      # 显示进程ID
      echo "Started process with PID $! - Log file: ${log_file}"
      
      # 给每个任务一些启动时间，防止资源竞争
      sleep 2
    done
  done
done

echo "所有interval ${interval}实验已启动。请在logs目录查看进度。"