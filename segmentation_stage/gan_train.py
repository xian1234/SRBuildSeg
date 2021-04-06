'''
source:https://github.com/xian1234/SRBuildSeg
'''
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from tqdm import tqdm

from time import time

from networks.unet import Unet, PSPNet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DeepLabV3
from framework import MyFrame
from loss import dice_bce_loss
from data_ganx4 import ImageFolder
import pdb

if __name__ == '__main__':

    SHAPE = (1024, 1024)
    ROOT = 'experiment/'
    training_root = 'img_x8/'
    trainlist = os.listdir(training_root)
    NAME = 'gan_pspnet_x8'
    BATCHSIZE_PER_CARD = 6

    #solver = MyFrame(DinkNet34, dice_bce_loss, 2e-3)
    solver = MyFrame(PSPNet, dice_bce_loss, 2e-3)

    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(trainlist, ROOT)
    a=dataset[1]
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    mylog = open('NAME+'.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = 80
    train_epoch_best_loss = 100.
    for epoch in tqdm(range(1, total_epoch + 1)):
        #pdb.set_trace()
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            if len(img)<=1 or len(mask)<=1:
                continue
            imgs = img.cuda()
            true_mask = mask.cuda()
            solver.set_input(imgs, true_mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss

            print('\r','training loss:{0}'.format(train_loss),end="",flush=True)
        train_epoch_loss /= len(data_loader_iter)
        print('********',file=mylog)
        print('epoch:',epoch,'   time:',int(time()-tic),file=mylog)
        print('train_loss:',train_epoch_loss,file=mylog)
        print('SHAPE:',SHAPE,file=mylog)
        print('********')
        print('epoch:',epoch,'    time:',int(time()-tic))
        print('train_loss:',train_epoch_loss)
        print('SHAPE:',SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save(NAME+'.th')
        if no_optim > 6:
            print('early stop at %d epoch' % epoch,file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(NAME+'.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()
