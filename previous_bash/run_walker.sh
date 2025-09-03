export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=10
set -x
set -e
# UNIT TEST

# stage 1
uncertainty=0
python -u ml/RL.py \
    --config configs/samples/ALR/walker2d_ALR.yml \
    --output ./save_policy/walker_${uncertainty}/stage1 \
    --uncertainty ${uncertainty}

python -u ml/svd_approximation.py \
    --policy_model_path ./save_policy/walker_${uncertainty}/stage1/Walker2dRandom-v0_epoch_695.pt \
    --output_model_path ./save_policy/walker_${uncertainty}/stage2/init.pt 

for ((i=2; i<epoch; i++))
do
j=$((i+1))
python -u ml/RL.py \
    --config configs/samples/ALR/walker2d_ALR.yml \
    --policy_model_path ./save_policy/walker_${uncertainty}/stage${i}/init.pt \
    --output ./save_policy/walker_${uncertainty}/stage${i} \
    --uncertainty ${uncertainty}

python -u ml/svd_approximation.py \
    --policy_model_path ./save_policy/walker_${uncertainty}/stage${i}/Walker2dRandom-v0_epoch_695.pt \
    --output_model_path ./save_policy/walker_${uncertainty}/stage${j}/init.pt \
    --uncertainty ${uncertainty}
done