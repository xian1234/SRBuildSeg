import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class CrossnetDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(CrossnetDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ_UX4, self.paths_Ref, self.paths_SR, self.paths_GT = None, None, None, None
        self.sizes_LQ_UX4, self.sizes_Ref, self.sizes_SR, self.sizes_GT = None, None, None, None
        self.LQ_UX4_env, self.Ref_env, self.SR_env, self.GT_env = None, None, None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_Ref, self.sizes_Ref = util.get_image_paths(self.data_type, opt['dataroot_Ref'])
        self.paths_SR, self.sizes_SR = util.get_image_paths(self.data_type, opt['dataroot_SR'])
        self.paths_LQ_UX4, self.sizes_LQ_UX4 = util.get_image_paths(self.data_type, opt['dataroot_LQ_UX4'])

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_Ref and self.paths_GT:
            assert len(self.paths_Ref) == len(
                self.paths_GT
            ), 'GT and Ref datasets have different number of images - {}, {}.'.format(
                len(self.paths_Ref), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_UX4_env = lmdb.open(self.opt['dataroot_LQ_UX4'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.Ref_env = lmdb.open(self.opt['dataroot_Ref'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.SR_env = lmdb.open(self.opt['dataroot_SR'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_UX4_env is None):
            self._init_lmdb()
        GT_path, LQ_UX4_path, Ref_path, SR_path = None, None, None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        #if self.opt['color']:  # change color space if necessary
        #    img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get Ref image
        Ref_path = self.paths_Ref[index]
        resolution = [int(s) for s in self.sizes_Ref[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_Ref = util.read_img(self.Ref_env, Ref_path, resolution)
        if self.opt['Ref_color']:  # change color space if necessary
            img_Ref = util.channel_convert(img_Ref.shape[2], self.opt['Ref_color'], [img_Ref])[0]

        SR_path = self.paths_SR[index]
        resolution = [int(s) for s in self.sizes_SR[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_SR = util.read_img(self.SR_env, SR_path, resolution)

        # get LQ_UX4 image
        if self.paths_LQ_UX4:
            LQ_UX4_path = self.paths_LQ_UX4[index]
            resolution = [int(s) for s in self.sizes_LQ_UX4[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ_UX4 = util.read_img(self.LQ_UX4_env, LQ_UX4_path, resolution)

        if self.opt['phase'] == 'train':
            H, W, C = img_LQ_UX4.shape
            LQ_size = GT_size

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ_UX4 = img_LQ_UX4[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_Ref = img_Ref[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_SR = img_SR[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_GT = img_GT[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation - flip, rotate
            img_LQ_UX4, img_Ref, img_SR, img_GT = util.augment([img_LQ_UX4, img_Ref, img_SR, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ_UX4 = img_LQ_UX4[:, :, [2, 1, 0]]
            img_SR = img_SR[:, :, [2, 1, 0]]
        if img_Ref.shape[2] == 3:
            img_Ref = img_Ref[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ_UX4 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_UX4, (2, 0, 1)))).float()
        img_SR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_SR, (2, 0, 1)))).float()
        img_Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Ref, (2, 0, 1)))).float()

        return {'LQ_UX4': img_LQ_UX4,'Ref': img_Ref, 'SR': img_SR, 'GT': img_GT, 'LQ_UX4_path': LQ_UX4_path, 'Ref_path': Ref_path, 'SR_path': SR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
