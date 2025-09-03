#!/bin/bash
export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=8
# 创建logs目录
mkdir -p logs

# 要测试的参数
uncertainty=0.5  # 固定uncertainty
betas=(0.98)
d_values=(300 500 700 1000)  # 测试不同的d值
seeds=(1 2 3)  # 减少到3个seeds

# 遍历不同的d值
for d in "${d_values[@]}"
do
  # 为每个d值创建对应的目录
  mkdir -p logs/d_${d}
  
  for beta in "${betas[@]}"
  do
    # 为每个beta值创建对应的目录
    mkdir -p save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}
    
    # 对每个组合运行不同的seed
    for seed in "${seeds[@]}"
    do
      echo "Running with d=${d}, uncertainty=${uncertainty}, beta=${beta}, seed=${seed}"
      
      # 创建日志文件名
      log_file="logs/d_${d}/walker_d_${d}_seed_${seed}.log"
      
      # 使用nohup运行实验并将输出重定向到日志文件
      nohup bash -c "
        set -x
        set -e
        
        # stage 1
        python -u ml/RL.py \
          --config configs/samples/ALR/walker2d_ALR.yml \
          --output ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage1 \
          --uncertainty ${uncertainty} \
          --d_step ${d}
        
        python -u ml/svd_approximation.py \
          --policy_model_path ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage1/Walker2dRandom-v0_epoch_695.pt \
          --output_model_path ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage2/init.pt \
          --beta ${beta}
        
        for ((i=2; i<${epoch}; i++))
        do
          j=\$((i+1))
          python -u ml/RL.py \
            --config configs/samples/ALR/walker2d_ALR.yml \
            --policy_model_path ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/init.pt \
            --output ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i} \
            --uncertainty ${uncertainty} \
            --d_step ${d}
          
          python -u ml/svd_approximation.py \
            --policy_model_path ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${i}/Walker2dRandom-v0_epoch_695.pt \
            --output_model_path ./save_policy/walker_d_${d}_uncertainty_${uncertainty}_beta_${beta}/seed_${seed}/stage\${j}/init.pt \
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

echo "所有d敏感度实验已启动。请在logs目录查看进度。"
