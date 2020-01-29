#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/eval_drill_down_3x128_rl.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Some notes on the parameters:
# rl_finetune: use in lib/modules/context_encoder.py, binary mode (rl_finetune > 0 or not)
# explore_mode: use in lib/modules/context_encoder.py 
# policy_mode: use in lib/modules/context_encoder.py
###############################################################
time python ./tools/eval_region.py --cuda --use_txt_context=True --num_workers=1 --instance_dim=3 --n_feature_dim=128 --rl_finetune=2 --explore_mode=5 --policy_mode=0 --final_loss_mode=3 --lr=2e-5 --exp_name=vg_f128_i3_rl_test --pretrained=vg_f128_i3_rl_ckpt
