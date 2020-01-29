
#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/train_logs/eval_R-RE_1280.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

###############################################################
# Some notes on the parameters:
# tirg_rnn: actually almost the same as GRU
###############################################################

python ./tools/eval_region.py --cuda --use_txt_context=False --paragraph_model=True --num_workers=1 --n_feature_dim=1280 --exp_name=vg_R-RE_1280_test --pretrained=vg_R-RE_f1280_ckpt
