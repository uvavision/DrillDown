#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/eval_R-HRE_1280.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Some notes on the parameters:
# tirg_rnn: actually almost the same as GRU
# loss_reduction_mode
###############################################################

python ./tools/eval_region.py --use_txt_context=True --num_workers=1 --loss_reduction_mode=1 --n_feature_dim=1280 --tirg_rnn=False --exp_name=vg_R-HRE_1280_test --pretrained=vg_f1280_R-HRE_ckpt