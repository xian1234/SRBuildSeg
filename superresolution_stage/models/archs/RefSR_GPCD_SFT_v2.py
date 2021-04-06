import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.net_utils import *
try:
    # from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
    from models.archs.DCNv2.dcn_v2 import DCN_sep as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class SRNTT(nn.Module):
    """
    PyTorch Module for SRNTT.
    Now x4 is only supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blucks : int, optional
        the number of residual blocks for each module.
    """
    def __init__(self, ngf=64, n_blocks=16, groups=8, use_weights=False):
        super(SRNTT, self).__init__()
        self.get_g_nopadding = Get_gradient_nopadding()
        self.Encoder = Encoder_2(3)
        self.Encoder_grad = Encoder_2(3)
        self.gpcd_align = PCD_Align(nf=ngf, groups=groups)
        self.content_extractor = ContentExtractor(ngf, n_blocks)
        self.texture_transfer = TextureTransfer(ngf, n_blocks, use_weights)
        #init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, LR, LR_UX4,SR, Ref,Ref_DUX4, weights=None):
        """
        Parameters
        ---
        x : torch.Tensor
            the input image of SRNTT.
        maps : dict of torch.Tensor
            the swapped feature maps on relu3_1, relu2_1 and relu1_1.
            depths of the maps are 256, 128 and 64 respectively.
        """
        LR_UX4_grad = self.get_g_nopadding(LR_UX4)
        Ref_DUX4_grad = self.get_g_nopadding(Ref_DUX4)
        Ref_grad = self.get_g_nopadding(Ref)


        LR_conv1, LR_conv2, LR_conv3 = self.Encoder(LR_UX4)        
        HR1_conv1, HR1_conv2, HR1_conv3 = self.Encoder(Ref_DUX4)
        HR2_conv1, HR2_conv2, HR2_conv3 = self.Encoder(Ref)
        
        #grad
        LR_conv1_grad, LR_conv2_grad, LR_conv3_grad = self.Encoder_grad(LR_UX4_grad)
        HR1_conv1_grad, HR1_conv2_grad, HR1_conv3_grad = self.Encoder_grad(Ref_DUX4_grad)
        HR2_conv1_grad, HR2_conv2_grad, HR2_conv3_grad = self.Encoder_grad(Ref_grad)

        LR_conv1 = torch.cat([LR_conv1,LR_conv1_grad], dim=1)
        LR_conv2 = torch.cat([LR_conv2,LR_conv2_grad], dim=1)
        LR_conv3 = torch.cat([LR_conv3,LR_conv3_grad], dim=1)
        HR2_conv1 = torch.cat([HR2_conv1,HR2_conv1_grad], dim=1)
        HR2_conv2 = torch.cat([HR2_conv2,HR2_conv2_grad], dim=1)
        HR2_conv3 = torch.cat([HR2_conv3,HR2_conv3_grad], dim=1)
        HR1_conv1 = torch.cat([HR1_conv1,HR1_conv1_grad], dim=1)
        HR1_conv2 = torch.cat([HR1_conv2,HR1_conv2_grad], dim=1)
        HR1_conv3 = torch.cat([HR1_conv3,HR1_conv3_grad], dim=1)

        LR_fea_l = [LR_conv1, LR_conv2, LR_conv3]
        Ref_use_fea_l = [HR2_conv1, HR2_conv2, HR2_conv3]
        Ref_fea_l = [HR1_conv1, HR1_conv2, HR1_conv3]

        Ref_conv1, Ref_conv2, Ref_conv3 = self.gpcd_align(Ref_use_fea_l,Ref_fea_l,LR_fea_l)
        maps = [Ref_conv1, Ref_conv2, Ref_conv3]

        base = F.interpolate(LR, None, 4, 'bilinear', False)
        upscale_plain, content_feat = self.content_extractor(LR)

        upscale_srntt = self.texture_transfer(
                    content_feat, maps)
        return upscale_srntt + base


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf * 2 , nf * 2, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf*2, nf*2, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L3_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf*2, nf*2, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf * 2, nf * 2, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 3, nf * 2, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf * 2, nf * 2, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.cas_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, Ref_use_fea_l, Ref_fea_l, LR_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        Ref_fea_l, LR_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([Ref_fea_l[2], LR_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.L3_dcnpack([Ref_use_fea_l[2], L3_offset])
        L3_fea_output = self.lrelu(self.L3_fea_conv(L3_fea))
        # L2
        L2_offset = torch.cat([Ref_fea_l[1], LR_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([Ref_use_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_output = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        # L1
        L1_offset = torch.cat([Ref_fea_l[0], LR_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([Ref_use_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea_output, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, LR_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea_output = self.cas_dcnpack([L1_fea, offset])
        L1_fea_output = self.lrelu(self.cas_fea_conv(L1_fea_output))

        return L1_fea_output, L2_fea_output, L3_fea_output


class ContentExtractor(nn.Module):
    """
    Content Extractor for SRNTT, which outputs maps before-and-after upscale.
    more detail: https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L73.
    Currently this module only supports `scale_factor=4`.

    Parameters
    ---
    ngf : int, optional
        a number of generator's features.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h


class TextureTransfer(nn.Module):
    """
    Conditional Texture Transfer for SRNTT,
        see https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L116.
    This module is devided 3 parts for each scales.

    Parameters
    ---
    ngf : int
        a number of generator's filters.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16, use_weights=False):
        super(TextureTransfer, self).__init__()

        # for small scale
        self.sft_head_small = SFTLayer()      

        self.body_small = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for medium scale
        self.sft_head_medium = SFTLayer()

        self.body_medium = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for large scale
        self.sft_head_large = SFTLayer()

        self.body_large = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )

        if use_weights:
            self.a = nn.Parameter(torch.ones(3), requires_grad=True)
            self.b = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, x, maps, weights=None):
        # compute weighted maps
        #if hasattr(self, 'a') and weights is not None:
        #    for idx, layer in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
        #        weights_scaled = F.interpolate(
        #            F.pad(weights, (1, 1, 1, 1), mode='replicate'),
        #            scale_factor=2**idx,
        #            mode='bicubic',
        #            align_corners=True) * self.a[idx] + self.b[idx]
        #        maps[layer] *= torch.sigmoid(weights_scaled)

        # small scale
        h = self.sft_head_small(x, maps[2])
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # medium scale
        h = self.sft_head_medium(x, maps[1])
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # large scale
        h = self.sft_head_large(x, maps[0])
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x


class ResBlock(nn.Module):
    """
    Basic residual block for SRNTT.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class SFTLayer(nn.Module):
    def __init__(self, nf=64, n_condition=64):
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




if __name__ == "__main__":
    device = torch.device('cuda:0')

    x = torch.rand(16, 3, 24, 24).to(device)

    maps = {}
    maps.update({'relu3_1': torch.rand(16, 256, 24, 24).to(device)})
    maps.update({'relu2_1': torch.rand(16, 128, 48, 48).to(device)})
    maps.update({'relu1_1': torch.rand(16, 64, 96, 96).to(device)})

    model = SRNTT().to(device)
    _, out = model(x, maps)
