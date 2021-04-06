#!/bin/sh

ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python train.py -opt options_rs/train_eegan.yml
