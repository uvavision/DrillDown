#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/train_R-RankFusion_256.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_region.py --cuda --use_txt_context=False --num_workers=1 --n_feature_dim=256 --tirg_rnn=False --exp_name=vg_R-RankFusion_256