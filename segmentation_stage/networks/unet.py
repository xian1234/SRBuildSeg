import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from collections import OrderedDict
import math
import torchvision

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.down1 = self.conv_stage(3, 8)
        self.down2 = self.conv_stage(8, 16)
        self.down3 = self.conv_stage(16, 32)
        self.down4 = self.conv_stage(32, 64)
        self.down5 = self.conv_stage(64, 128)
        self.down6 = self.conv_stage(128, 256)
        self.down7 = self.conv_stage(256, 512)
        
        self.center = self.conv_stage(512, 1024)
        #self.center_res = self.resblock(1024)
        
        self.up7 = self.conv_stage(1024, 512)
        self.up6 = self.conv_stage(512, 256)
        self.up5 = self.conv_stage(256, 128)
        self.up4 = self.conv_stage(128, 64)
        self.up3 = self.conv_stage(64, 32)
        self.up2 = self.conv_stage(32, 16)
        self.up1 = self.conv_stage(16, 8)
        
        self.trans7 = self.upsample(1024, 512)
        self.trans6 = self.upsample(512, 256)
        self.trans5 = self.upsample(256, 128)
        self.trans4 = self.upsample(128, 64)
        self.trans3 = self.upsample(64, 32)
        self.trans2 = self.upsample(32, 16)
        self.trans1 = self.upsample(16, 8)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.max_pool = nn.MaxPool2d(2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU()
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )
    
    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))
        conv5_out = self.down5(self.max_pool(conv4_out))
        conv6_out = self.down6(self.max_pool(conv5_out))
        conv7_out = self.down7(self.max_pool(conv6_out))
        
        out = self.center(self.max_pool(conv7_out))
        #out = self.center_res(out)

        out = self.up7(torch.cat((self.trans7(out), conv7_out), 1))
        out = self.up6(torch.cat((self.trans6(out), conv6_out), 1))
        out = self.up5(torch.cat((self.trans5(out), conv5_out), 1))
        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))

        out = self.conv_last(out)

        return out

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class NaivePyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim):
        super(NaivePyramidPoolingModule, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        out.append(F.interpolate(self.feature(x),
                                 x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, layers=18, bins=(1, 2, 3, 6), dropout=0.1, zoom_factor=8, use_ppm=True,
                 naive_ppm=False, criterion=nn.CrossEntropyLoss(ignore_index=0), pretrained=True):
        super(PSPNet, self).__init__()

        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        self.layers = layers
        if layers == 18:
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == 50:
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        else:
            resnet = torchvision.models.resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # fea_dim = resnet.layer4[-1].conv3.out_channels
        fea_dim = 512
        if layers == 18:
            for n, m in self.layer3[0].named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                if 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4[0].named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                if 'downsample.0' in n:
                    m.stride = (1, 1)

        else:
            fea_dim = 2048
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if use_ppm:
            if naive_ppm:
                self.ppm = NaivePyramidPoolingModule(fea_dim, int(fea_dim))
            else:
                self.ppm = PyramidPoolingModule(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls_head = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.size()
        #assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = x_size[2]
        w = x_size[3]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # use different layers output as auxiliary supervision
        # the reception field of layer3 in Res18 is too small, may influence the
        # segmentation result, so we use the layer4 output for auxiliary supervision.
        if self.layers != 18:
            feat_tmp= self.layer3(x)
            feat = self.layer4(feat_tmp)
            feat = self.ppm(feat)
        else:
            x = self.layer3(x)
            feat_tmp = self.layer4(x)
            feat = self.ppm(feat_tmp)

        pred = self.cls_head(feat)
        if self.zoom_factor != 1:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        return F.sigmoid(pred)