import argparse
import glob
import logging
import math
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.eegan as eegan
import options.options as option
import scipy.io as sio
from data import create_dataloader, create_dataset
from utils import util

from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--model_SPSR_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    #parser.add_argument('--model_P_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    return parser

################################################################################
args = get_parser().parse_args()
model_SPSR_path = args.model_SPSR_path
exp_name = args.exp_name
dataset = args.dataset
save_path = args.save_path
scale = 4

print(dataset, exp_name)

model = eegan.EEGAN()
model.load_state_dict(torch.load(model_SPSR_path), strict=True)
model.eval()
model = model.cuda()

#if dataset == 'Gaofen1':
#    img_list = glob.glob('../../dataset/val_rs/200309_GF-1/subimgs/*')
#elif dataset == 'val_1st':
#    img_list = glob.glob('/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_1st/LR/*')
#    HR_path = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_1st/HR'
#elif dataset == 'val_2nd':
#    img_list = glob.glob('/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_2nd/LR/*')
#    HR_path = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_2nd/HR'
#elif dataset == 'val_3rd':
#    img_list = glob.glob('/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_3rd/LR/*')
#    HR_path = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_3rd/HR'
#elif dataset == 'val_4th':
#    img_list = glob.glob('/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_4th/LR/*')
#    HR_path = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/RefSR/github_pub_round1/dataset/val/val_4th/HR'
img_list = glob.glob('/data/zlx/rfsr/github_pub_round1/dataset/val/val_1st/LR/*')
HR_path = '/data/zlx/rfsr/github_pub_round1/dataset/val/val_1st/HR'

util.mkdir_and_rename(osp.join(save_path,'{}_exp{}'.format(dataset, exp_name)))

PSNR_avg = 0
SSIM_avg = 0

stat_time = 0

for ii, img_path in enumerate(sorted(img_list)):
    # if '160' in img_path:
    base_name = osp.splitext(osp.basename(img_path))[0]
    use_name = base_name + '.jpg'

    img_GT = cv2.imread(osp.join(HR_path,use_name))

    img_LR = cv2.imread(img_path) / 255.
    #img_LR = img_LR[:, :, [2, 1, 0]]

    img_LR = torch.from_numpy(
        np.ascontiguousarray(np.transpose(
            img_LR, (2, 0, 1)))).float().unsqueeze(0).cuda()
    
    with torch.no_grad():
        begin_time = time.time()
        frame,sr_base,output= model(img_LR)
        end_time = time.time()
        stat_time += (end_time-begin_time)
        #print(end_time-begin_time)

    output = util.tensor2img(output.squeeze(0))
    frame  = util.tensor2img(frame.squeeze(0))
    sr_base = util.tensor2img(sr_base.squeeze(0))

    # save images
    save_path_name = osp.join(save_path,'{}_exp{}/{}.png'.format(
        dataset, exp_name, base_name))
    merge = np.concatenate((frame, sr_base, output), axis=1)
    util.save_img(merge, save_path_name)
  
    # calculate PSNR
    sr_img, gt_img = util.crop_border([output, img_GT], scale)
    PSNR_avg += util.calculate_psnr(sr_img, gt_img)
    SSIM_avg += util.calculate_ssim(sr_img, gt_img)

print('average PSNR: ', PSNR_avg / len(img_list))
print('average SSIM: ', SSIM_avg / len(img_list))
print('time: ', stat_time / len(img_list))
