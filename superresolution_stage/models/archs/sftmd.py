""" Architecture for SFTMD """
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch.nn.utils.spectral_norm as spectral_norm


class SFTLayer(nn.Module):
    def __init__(self, nf=64, n_condition=10):
        super(SFTLayer, self).__init__()
        # TODO: can use shared convolution layers to save computation
        self.mul_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add


class SFTLayer_SN(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1, bias_sn=False):
        super(SFTLayer_SN, self).__init__()
        # TODO: can use shared convolution layers to save computation
        self.mul_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.mul_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)
        self.add_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.add_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)
        if bias_sn:
            self.mul_conv1 = spectral_norm(self.mul_conv1, name='bias',
                                           n_power_iterations=n_power_iterations)
            self.mul_conv2 = spectral_norm(self.mul_conv2, name='bias',
                                           n_power_iterations=n_power_iterations)
            self.add_conv1 = spectral_norm(self.add_conv1, name='bias',
                                           n_power_iterations=n_power_iterations)
            self.add_conv2 = spectral_norm(self.add_conv2, name='bias',
                                           n_power_iterations=n_power_iterations)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return features * mul + add


class SFTLayer_SN_Norm(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1, norm='batch'):
        super(SFTLayer_SN_Norm, self).__init__()
        # TODO: can use shared convolution layers to save computation
        if norm == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True,
                                           track_running_stats=True)

        self.mul_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.mul_norm1 = norm_layer(num_features=32)
        self.mul_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)
        self.mul_norm2 = norm_layer(num_features=nf)
        self.add_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.add_norm1 = norm_layer(num_features=32)
        self.add_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)
        self.add_norm2 = norm_layer(num_features=nf)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(
            self.mul_norm2(self.mul_conv2(self.lrelu(self.mul_norm1(self.mul_conv1(cat_input))))))
        add = self.add_norm2(self.add_conv2(self.lrelu(self.add_norm1(self.add_conv1(cat_input)))))
        return features * mul + add


class SFTLayer_SN_ReLU(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1):
        super(SFTLayer_SN_ReLU, self).__init__()
        # TODO: can use shared convolution layers to save computation
        self.mul_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.mul_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)
        self.add_conv1 = spectral_norm(
            nn.Conv2d(nf + n_condition, 32, kernel_size=3, stride=1, padding=1), name='weight',
            n_power_iterations=n_power_iterations)
        self.add_conv2 = spectral_norm(nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1),
                                       name='weight', n_power_iterations=n_power_iterations)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, conditions):
        cat_input = torch.cat((features, conditions), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.relu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.relu(self.add_conv1(cat_input)))
        return features * mul + add


class SFTResidualBlock(nn.Module):
    def __init__(self, nf=64, n_condition=10):
        super(SFTResidualBlock, self).__init__()
        self.sft1 = SFTLayer(nf=nf, n_condition=n_condition)
        self.sft2 = SFTLayer(nf=nf, n_condition=n_condition)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, features, conditions):
        fea = self.lrelu(self.sft1(features, conditions))
        fea = self.lrelu(self.sft2(self.conv1(fea), conditions))
        fea = self.conv2(fea)
        return features + fea

class SFTResidualBlock_SN(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1, bias_sn=False):
        super(SFTResidualBlock_SN, self).__init__()
        self.sft1 = SFTLayer_SN(nf=nf, n_condition=n_condition)
        self.sft2 = SFTLayer_SN(nf=nf, n_condition=n_condition)
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)
        self.conv2 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if bias_sn:
            self.conv1 = spectral_norm(self.conv1, name='bias',
                                       n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(self.conv2, name='bias',
                                       n_power_iterations=n_power_iterations)

        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, features, conditions):
        fea = self.lrelu(self.sft1(features, conditions))
        fea = self.lrelu(self.sft2(self.conv1(fea), conditions))
        fea = self.conv2(fea)
        return features + fea


class SFTResidualBlock_SN_Norm(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1, norm='batch'):
        super(SFTResidualBlock_SN_Norm, self).__init__()
        if norm == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True,
                                           track_running_stats=True)

        self.sft1 = SFTLayer_SN_Norm(nf=nf, n_condition=n_condition,
                                     n_power_iterations=n_power_iterations, norm=norm)
        self.sft2 = SFTLayer_SN_Norm(nf=nf, n_condition=n_condition,
                                     n_power_iterations=n_power_iterations, norm=norm)
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)
        self.norm1 = norm_layer(num_features=64)
        self.conv2 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)
        self.norm2 = norm_layer(num_features=64)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, features, conditions):
        fea = self.lrelu(self.sft1(features, conditions))
        fea = self.lrelu(self.sft2(self.norm1(self.conv1(fea)), conditions))
        fea = self.norm2(self.conv2(fea))
        return features + fea


class SFTResidualBlock_SN_ReLU(nn.Module):
    def __init__(self, nf=64, n_condition=10, n_power_iterations=1):
        super(SFTResidualBlock_SN_ReLU, self).__init__()
        self.sft1 = SFTLayer_SN_ReLU(nf=nf, n_condition=n_condition)
        self.sft2 = SFTLayer_SN_ReLU(nf=nf, n_condition=n_condition)
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)
        self.conv2 = spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   name='weight', n_power_iterations=n_power_iterations)

        self.relu = nn.ReLU(inplace=True)

        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, features, conditions):
        fea = self.relu(self.sft1(features, conditions))
        fea = self.relu(self.sft2(self.conv1(fea), conditions))
        fea = self.conv2(fea)
        return features + fea


class SFTMD(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16):
        super(SFTMD, self).__init__()
        self.n_RB = n_RB

        self.conv_first = nn.Conv2d(inc, nf, 3, stride=1, padding=1)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i), SFTResidualBlock(nf=nf, n_condition=n_condition))

        self.sft_extra = SFTLayer(nf=nf, n_condition=n_condition)
        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code, spatial=False, extra=False):
        _, _, H, W = input.size()
        if not spatial:
            Bk, Ck = kernel_code.size()
            kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, kernel_code)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, kernel_code)))

        out = self.conv_final(self.upscale(fea))
        if extra:
            return out, fea
        else:
            return out

class SFTMD_Ushape(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16):
        super(SFTMD_Ushape, self).__init__()
        self.n_RB = n_RB

        self.conv_first = nn.Conv2d(inc, nf, 3, stride=1, padding=1)

        # downsample operation
        for i in range(n_RB // 2):
            self.add_module('SFTRB_down' + str(i), SFTResidualBlock(nf=nf, n_condition=n_condition))

        self.mid_layer = SFTResidualBlock(nf=nf, n_condition=n_condition)
        # upsample operation
        for i in range(n_RB // 2):
            self.add_module('SFTRB_up' + str(i), SFTResidualBlock(nf=nf, n_condition=n_condition))

        self.sft_extra = SFTLayer(nf=nf, n_condition=n_condition)
        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, input, kernel_code, spatial=False, extra=False):
        _, _, H_in, W_in = input.size()
        kernel_code_ori = kernel_code.clone()
        # if not spatial:
        #     Bk, Ck = kernel_code_ori.size()
        #     kernel_code = kernel_code_ori.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        Bk, Ck = kernel_code_ori.size()

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()

        # down_scale
        kernel_code_list = []
        for i in range(self.n_RB // 2):
            H = int(H_in * 2 ** (-1 * i))
            W = int(W_in * 2 ** (-1 * i))
            kernel_code = kernel_code_ori.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))
            fea_sft_x2 = self.__getattr__('SFTRB_down' + str(i))(fea_sft, kernel_code)

            fea_sft = self.max_pool(fea_sft_x2)
            kernel_code_list.insert(0, kernel_code)

        H = int(H_in * 2 ** (-1 * (self.n_RB // 2)))
        W = int(W_in * 2 ** (-1 * (self.n_RB // 2)))
        kernel_code = kernel_code_ori.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))
        fea_sft = self.mid_layer(fea_sft, kernel_code)

        #up_scale
        for i in range(self.n_RB // 2):
            fea_sft = F.interpolate(fea_sft, scale_factor=2, mode='bilinear', align_corners=False)
            fea_sft = self.__getattr__('SFTRB_up' + str(i))(fea_sft, kernel_code_list[i])

        kernel_code = kernel_code_list[self.n_RB // 2 - 1]
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, kernel_code)))

        out = self.conv_final(self.upscale(fea))
        if extra:
            return out, fea
        else:
            return out


class SFTMD_Noise_JPEG(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=12, scale=4, n_RB=16):
        super(SFTMD_Noise_JPEG, self).__init__()
        self.n_RB = n_RB

        self.conv_first = nn.Conv2d(inc, nf, 3, stride=1, padding=1)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i), SFTResidualBlock(nf=nf, n_condition=n_condition))

        self.sft_extra = SFTLayer(nf=nf, n_condition=n_condition)
        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code, noise, jpeg, spatial=False, extra=False):
        _, _, H, W = input.size()
        if not spatial:
            codes = torch.cat((kernel_code, noise, jpeg), dim=1)
            Bk, Ck = codes.size()
            codes = codes.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, codes)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, codes)))

        out = self.conv_final(self.upscale(fea))
        if extra:
            return out, fea
        else:
            return out


class SFTMD_SN_Noise_JPEG(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16, n_power_iterations=1,
                 norm=None, bias_sn=False):
        super(SFTMD_SN_Noise_JPEG, self).__init__()
        self.n_RB = n_RB
        if bias_sn:
            print('Bias SN')

        self.conv_first = spectral_norm(nn.Conv2d(inc, nf, 3, stride=1, padding=1), name='weight',
                                        n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_first = spectral_norm(self.conv_first, name='bias',
                                            n_power_iterations=n_power_iterations)
        for i in range(n_RB):
            if norm is None:
                self.add_module('SFTRB' + str(i),
                                SFTResidualBlock_SN(nf=nf, n_condition=n_condition, bias_sn=False))
            else:
                self.add_module(
                    'SFTRB' + str(i),
                    SFTResidualBlock_SN_Norm(nf=nf, n_condition=n_condition,
                                             n_power_iterations=n_power_iterations, norm=norm))
        if norm is None:
            self.sft_extra = SFTLayer_SN(nf=nf, n_condition=n_condition, bias_sn=False)
        else:
            self.sft_extra = SFTLayer_SN_Norm(nf=nf, n_condition=n_condition,
                                              n_power_iterations=n_power_iterations, norm=norm)
        self.conv_extra = spectral_norm(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_extra = spectral_norm(self.conv_extra, name='bias',
                                            n_power_iterations=n_power_iterations)
        if scale == 4:
            if bias_sn:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            else:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
        else:
            if bias_sn:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            else:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.1, inplace=True),
                )

        self.conv_final = spectral_norm(
            nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_final = spectral_norm(self.conv_final, name='bias',
                                            n_power_iterations=n_power_iterations)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code, noise, jpeg, spatial=False, extra=False):
        _, _, H, W = input.size()
        if not spatial:
            codes = torch.cat((kernel_code, noise, jpeg), dim=1)
            Bk, Ck = codes.size()
            codes = codes.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, codes)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, codes)))

        out = self.conv_final(self.upscale(fea))
        if extra:
            return out, fea
        else:
            return out


class SFTMD_SN(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16, n_power_iterations=1,
                 norm=None, bias_sn=False):
        super(SFTMD_SN, self).__init__()
        self.n_RB = n_RB
        if bias_sn:
            print('Bias SN')

        self.conv_first = spectral_norm(nn.Conv2d(inc, nf, 3, stride=1, padding=1), name='weight',
                                        n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_first = spectral_norm(self.conv_first, name='bias',
                                            n_power_iterations=n_power_iterations)
        for i in range(n_RB):
            if norm is None:
                self.add_module('SFTRB' + str(i), SFTResidualBlock_SN(nf=nf,
                                                                      n_condition=n_condition, bias_sn=False))
            else:
                self.add_module(
                    'SFTRB' + str(i),
                    SFTResidualBlock_SN_Norm(nf=nf, n_condition=n_condition,
                                             n_power_iterations=n_power_iterations, norm=norm))
        if norm is None:
            self.sft_extra = SFTLayer_SN(nf=nf, n_condition=n_condition, bias_sn=False)
        else:
            self.sft_extra = SFTLayer_SN_Norm(nf=nf, n_condition=n_condition,
                                              n_power_iterations=n_power_iterations, norm=norm)
        self.conv_extra = spectral_norm(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_extra = spectral_norm(self.conv_extra, name='bias',
                                            n_power_iterations=n_power_iterations)
        if scale == 4:
            if bias_sn:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            else:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
        else:
            if bias_sn:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        spectral_norm(
                            nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1,
                                      bias=True), name='weight',
                            n_power_iterations=n_power_iterations), name='bias',
                        n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            else:
                self.upscale = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                        name='weight', n_power_iterations=n_power_iterations),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.1, inplace=True),
                )

        self.conv_final = spectral_norm(
            nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)
        if bias_sn:
            self.conv_final = spectral_norm(self.conv_final, name='bias',
                                            n_power_iterations=n_power_iterations)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code, spatial=False, extra=False):
        _, _, H, W = input.size()
        if not spatial:
            Bk, Ck = kernel_code.size()
            kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, kernel_code)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, kernel_code)))

        out = self.conv_final(self.upscale(fea))
        if extra:
            return out, fea
        else:
            return out


class SFTMD_SN_Dropout(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16, n_power_iterations=1,
                 norm=None, dropSN=True):
        super(SFTMD_SN_Dropout, self).__init__()
        self.n_RB = n_RB

        self.conv_first = spectral_norm(nn.Conv2d(inc, nf, 3, stride=1, padding=1), name='weight',
                                        n_power_iterations=n_power_iterations)
        for i in range(n_RB):
            if norm is None:
                self.add_module('SFTRB' + str(i), SFTResidualBlock_SN(nf=nf,
                                                                      n_condition=n_condition))
            else:
                self.add_module(
                    'SFTRB' + str(i),
                    SFTResidualBlock_SN_Norm(nf=nf, n_condition=n_condition,
                                             n_power_iterations=n_power_iterations, norm=norm))
        if norm is None:
            self.sft_extra = SFTLayer_SN(nf=nf, n_condition=n_condition)
        else:
            self.sft_extra = SFTLayer_SN_Norm(nf=nf, n_condition=n_condition,
                                              n_power_iterations=n_power_iterations, norm=norm)

        if dropSN:
            self.conv_extra = spectral_norm(
                nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
                n_power_iterations=n_power_iterations)
            self.conv_extra2 = spectral_norm(
                nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
                n_power_iterations=n_power_iterations)
        else:
            self.conv_extra = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv_extra2 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

        if scale == 4:
            self.upscale = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(
                    nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = spectral_norm(
            nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code):
        _, _, H, W = input.size()
        Bk, Ck = kernel_code.size()
        kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, kernel_code)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, kernel_code)))
        fea = self.dropout(fea)
        fea = self.conv_extra2(fea)

        out = self.conv_final(self.upscale(fea))
        return out


class SFTMD_SN_ReLU(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16):
        super(SFTMD_SN_ReLU, self).__init__()
        self.n_RB = n_RB
        n_power_iterations = 1

        self.conv_first = spectral_norm(nn.Conv2d(inc, nf, 3, stride=1, padding=1), name='weight',
                                        n_power_iterations=n_power_iterations)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i),
                            SFTResidualBlock_SN_ReLU(nf=nf, n_condition=n_condition))

        self.sft_extra = SFTLayer_SN_ReLU(nf=nf, n_condition=n_condition)
        self.conv_extra = spectral_norm(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)
        if scale == 4:
            self.upscale = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale // 2),
                nn.ReLU(inplace=True),
                spectral_norm(
                    nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale // 2),
                nn.ReLU(inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                    name='weight', n_power_iterations=n_power_iterations),
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True),
            )

        self.conv_final = spectral_norm(
            nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True), name='weight',
            n_power_iterations=n_power_iterations)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, kernel_code):
        _, _, H, W = input.size()
        Bk, Ck = kernel_code.size()
        kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.relu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, kernel_code)
        fea = fea + fea_sft
        fea = self.conv_extra(self.relu(self.sft_extra(fea, kernel_code)))

        out = self.conv_final(self.upscale(fea))
        return out


class SFTMD_concat(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16):
        super(SFTMD_concat, self).__init__()
        self.n_RB = n_RB

        self.conv_first = nn.Conv2d(n_condition + 3, nf, 3, stride=1, padding=1)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i), arch_util.ResidualBlock_noBN(nf=nf))

        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input, kernel_code):
        B, _, H, W = input.size()

        Bk, Ck = kernel_code.size()
        kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(torch.cat((input, kernel_code), 1)))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(fea))

        out = self.conv_final(self.upscale(fea))
        return out


class SFTMD_kernel(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16, k=11):
        super(SFTMD_kernel, self).__init__()
        self.n_RB = n_RB

        self.fc_share_1 = nn.Linear(32, 100)
        self.fc_share_2 = nn.Linear(100, 200)
        self.fc_share_3 = nn.Linear(200, 400)
        self.fc_share_4 = nn.Linear(400, 200)

        self.fc_share_conv1_1 = nn.Linear(200, 200)
        self.fc_share_conv1_2 = nn.Linear(200, 10 * 3 * k * 1)
        self.fc_share_conv2_1 = nn.Linear(200, 200)
        self.fc_share_conv2_2 = nn.Linear(200, 10 * 10 * k * 1)

        self.conv_first = nn.Conv2d(10, nf, 3, stride=1, padding=1)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i), arch_util.ResidualBlock_noBN(nf=nf))

        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pad = (k - 1) // 2
        self.k = k

    def forward(self, input, kernel_code):
        B, _, H, W = input.size()

        # generate conv code
        kernel_code = kernel_code.view((B, -1))
        kernel_code = self.lrelu(self.fc_share_1(kernel_code))
        kernel_code = self.lrelu(self.fc_share_2(kernel_code))
        kernel_code = self.lrelu(self.fc_share_3(kernel_code))
        kernel_code = self.lrelu(self.fc_share_4(kernel_code))
        conv1_weight = self.fc_share_conv1_2(self.lrelu(self.fc_share_conv1_1(kernel_code)))
        conv2_weight = self.fc_share_conv2_2(self.lrelu(self.fc_share_conv2_1(kernel_code)))

        conv1_weight = conv1_weight.view((10, 3, self.k, 1))
        conv2_weight = conv2_weight.view((10, 10, 1, self.k))

        fea = self.lrelu(F.conv2d(input, conv1_weight, padding=(self.pad, 0)))
        fea = self.lrelu(F.conv2d(fea, conv2_weight, padding=(0, self.pad)))

        fea = self.lrelu(self.conv_first(fea))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(fea))

        out = self.conv_final(self.upscale(fea))
        return out


class SFTMD_coderefine(nn.Module):
    def __init__(self, inc=3, nf=64, n_condition=10, scale=4, n_RB=16):
        super(SFTMD_coderefine, self).__init__()
        self.n_RB = n_RB

        self.conv_first = nn.Conv2d(inc, nf, 3, stride=1, padding=1)
        for i in range(n_RB):
            self.add_module('SFTRB' + str(i), SFTResidualBlock(nf=nf, n_condition=n_condition))

        self.sft_extra = SFTLayer(nf=nf, n_condition=n_condition)
        self.conv_extra = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf, nf * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(nf, nf * scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.conv_final = nn.Conv2d(nf, inc, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fc1 = nn.Linear(n_condition, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, n_condition)

    def forward(self, input, kernel_code):
        _, _, H, W = input.size()
        kernel_code = self.lrelu(self.fc1(kernel_code))
        kernel_code = self.lrelu(self.fc2(kernel_code))
        kernel_code = self.lrelu(self.fc3(kernel_code))
        kernel_code = self.fc4(kernel_code)

        Bk, Ck = kernel_code.size()
        kernel_code = kernel_code.view((Bk, Ck, 1, 1)).expand((Bk, Ck, H, W))

        fea = self.lrelu(self.conv_first(input))
        fea_sft = fea.clone()
        for i in range(self.n_RB):
            fea_sft = self.__getattr__('SFTRB' + str(i))(fea_sft, kernel_code)
        fea = fea + fea_sft
        fea = self.conv_extra(self.lrelu(self.sft_extra(fea, kernel_code)))

        out = self.conv_final(self.upscale(fea))
        return out


class Corrector(nn.Module):
    def __init__(self, inc=3, n_condition=10, nf=64, conv_merge=True, use_bias=True):
        super(Corrector, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(inc, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(n_condition, nf, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Linear(nf, nf, bias=use_bias),
        ])

        if conv_merge:
            self.global_dense = nn.Sequential(*[
                nn.Conv2d(nf * 2, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
            ])

        self.nf = nf
        self.conv_merge = conv_merge

        self.fc1 = nn.Linear(nf, nf, bias=True)
        self.fc2 = nn.Linear(nf, nf, bias=True)
        self.fc3 = nn.Linear(nf, n_condition, bias=True)
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.lrelu = nn.LeakyReLU(0.1, True)

    def forward(self, input, code):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()  # LR_size

        code_ori = self.code_dense(code)
        if self.conv_merge:
            conv_code = code_ori.view((B, self.nf, 1, 1)).expand((B, self.nf, H_f, W_f))
            conv_mid = torch.cat((conv_input, conv_code), dim=1)
            conv_input = self.global_dense(conv_mid)

        fea = self.globalpooling(conv_input).view(conv_input.size(0), -1)
        fea = self.lrelu(self.fc1(fea))
        fea = self.lrelu(self.fc2(fea))
        out = self.fc3(fea)

        return out + code


class CorrectorV2(nn.Module):
    def __init__(self, inc=3, n_condition=10, nf=64, conv_merge=False, use_bias=True):
        super(CorrectorV2, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(inc, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(n_condition, nf, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Linear(nf, nf, bias=use_bias),
        ])

        if conv_merge:
            self.global_dense = nn.Sequential(*[
                nn.Conv2d(nf * 2, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
                nn.LeakyReLU(0.1, True),
            ])

        self.nf = nf
        self.conv_merge = conv_merge

        self.fc1 = nn.Linear(nf, nf, bias=True)
        self.fc2 = nn.Linear(nf, nf, bias=True)
        self.fc3 = nn.Linear(nf, n_condition, bias=True)
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.lrelu = nn.LeakyReLU(0.1, True)

    def forward(self, input, code):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()  # LR_size

        code_ori = self.code_dense(code)
        if self.conv_merge:
            conv_code = code_ori.view((B, self.nf, 1, 1)).expand((B, self.nf, H_f, W_f))
            conv_mid = torch.cat((conv_input, conv_code), dim=1)
            conv_input = self.global_dense(conv_mid)

        fea = self.globalpooling(conv_input).view(conv_input.size(0), -1)
        fea = self.lrelu(self.fc1(fea))
        fea = self.lrelu(self.fc2(fea))
        out = self.fc3(fea)

        return out + code_ori
