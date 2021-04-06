#!/bin/sh

ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH


TASK_NUM=8
srun -p ad_rs -n${TASK_NUM} --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=DBPN --kill-on-bad-exit=1 \
python -u train.py \
  -opt options_rs/train_dbpn.yml  \
  --launcher slurm


