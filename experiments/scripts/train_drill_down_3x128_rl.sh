#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/train_drill_down_3x128_rl.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Arguments:
# use_txt_context: use context encoder or not 
# rl_finetune: use in lib/modules/context_encoder.py, binary mode (rl_finetune > 0 or not)
# explore_mode: use in lib/modules/context_encoder.py 
# policy_mode: use in lib/modules/context_encoder.py
# pretrained: finetune from the supervised pretrained model.
#    The default model path is data/caches/region_ckpts/vg_f128_i3_sl_ckpt
###############################################################
time python ./tools/train_region.py --cuda --use_txt_context=True --num_workers=1 --instance_dim=3 --n_feature_dim=128 --rl_finetune=2 --explore_mode=5 --policy_mode=0 --final_loss_mode=3 --lr=2e-5 --exp_name=vg_rl_finetune_f128_i3 --pretrained=vg_f128_i3_sl_ckpt
