
#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/eval_R-RankFusion_1280.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Some notes on the parameters:
# tirg_rnn: actually almost the same as GRU
###############################################################

time python ./tools/eval_region.py --cuda --use_txt_context=False --num_workers=1 --n_feature_dim=256 --tirg_rnn=False  --exp_name=vg_R-RankFusion_256_test --rank_fusion=True --pretrained=vg_f256_rank_fusion_ckpt
