export PYTHONPATH=${PWD}:$PYTHONPATH
epoch=10
set -x
set -e
# UNIT TEST

# stage 1
uncertainty=0
python -u ml/RL.py \
    --config configs/samples/ALR/walker2d_ALR.yml \
    --output ./save_policy/walker_0/stage1 \
    --uncertainty 0