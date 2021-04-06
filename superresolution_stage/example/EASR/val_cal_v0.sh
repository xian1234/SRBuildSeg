ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

srun -p Test --gres=gpu:1 python -u val_cal_v0.py \
  --model_SPSR_path=/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/codes/example/DBPN/exp_v0/experiments/DBPN_v1_xiamen/models/125000_G.pth \
  --exp_name=ep125k \
  --dataset=val_4th \
  --save_path=/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/codes/example/DBPN/results_TGRS


