#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/train_R-HRE_1280.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Arguments:
# tirg_rnn: actually almost the same as GRU
###############################################################

python ./tools/train_region.py --cuda --use_txt_context=True --num_workers=1 --loss_reduction_mode=1 --n_feature_dim=1280 --tirg_rnn=False --exp_name=vg_R-HRE_1280 