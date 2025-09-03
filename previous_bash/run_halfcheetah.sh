export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=5
set -x
set -e

# 创建所需的目录结构
mkdir -p save_policy/halfcheetah

# UNIT TEST

# nohup ./run_halfcheetah.sh > halfcheetah_beta_98_seed1.log 2>&1 &

# stage 1

# python -u ml/RL.py \
#     --config configs/samples/ALR/halfcheetah.yml \
#     --output ./save_policy/halfcheetah/stage1 

# python -u ml/svd_approximation.py \
#     --policy_model_path ./save_policy/halfcheetah/stage1/HalfCheetah-v3_epoch_495.pt \
#     --output_model_path ./save_policy/halfcheetah/stage2/init.pt 

for i in $(seq 2 $((epoch-1))); do
    j=$((i+1))
    python -u ml/RL.py \
        --config configs/samples/ALR/halfcheetah.yml \
        --policy_model_path ./save_policy/halfcheetah/stage${i}/init.pt \
        --output ./save_policy/halfcheetah/stage${i}

    python -u ml/svd_approximation.py \
        --policy_model_path ./save_policy/halfcheetah/stage${i}/HalfCheetah-v3_epoch_495.pt \
        --output_model_path ./save_policy/halfcheetah/stage${j}/init.pt 
done