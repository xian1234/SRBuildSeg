ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python test.py \
  --model_SPSR_path=/data/zlx/xian/mmsr/example/eersgan/expx_2/experiments/eersgan_beijing/models/latest_G.pth \
  --exp_name=ep125k \
  --dataset=val_4th \
  --save_path=/data/zlx/xian/mmsr/example/eersgan/expx_2/


