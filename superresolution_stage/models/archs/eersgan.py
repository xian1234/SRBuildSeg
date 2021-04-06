import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import pdb
import torch.nn as nn
import torch.optim as optim
from models.archs.base_networks import *
from torchvision.transforms import *


#from layer import *
#from vgg19 import VGG19
def make_model(args, parent=False):
    return Net()


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        self.kernel_v = torch.from_numpy(np.array([[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]])).cuda().float()
        self.kernel_h = torch.from_numpy(np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]])).cuda().float()
        self.kernel_h = self.kernel_h.unsqueeze(0).unsqueeze(0)
        self.kernel_v = self.kernel_v.unsqueeze(0).unsqueeze(0)
        self.gradient_h = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.gradient_h.weight.data = self.kernel_h
        self.gradient_h.weight.requires_grad = False

        self.gradient_v = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.gradient_v.weight.data = self.kernel_v
        self.gradient_v.weight.requires_grad = False


    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = self.gradient_v(x_i.unsqueeze(1))
            x_i_h = self.gradient_h(x_i.unsqueeze(1))
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x

class Laplacian:
    def __init__(self):
        self.weight = torch.from_numpy(np.array([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]])).cuda().float()
        self.frame = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.frame.weight.data = self.weight
        self.frame.weight.requires_grad = False

    def __call__(self, x):
        out = self.frame(x)
        return out

class _Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(_Dense_Block, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=112, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))

        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))

        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))

        return cout8_dense


class EERSGAN(nn.Module):
    def __init__(self,num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=4):
        super(EERSGAN, self ).__init__() 
        self.scale = scale_factor
        self.lrelu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.denseblock1 = self.make_layer(_Dense_Block, 128)
        self.denseblock2 = self.make_layer(_Dense_Block, 256)
        self.denseblock3 = self.make_layer(_Dense_Block, 384)
        self.denseblock4 = self.make_layer(_Dense_Block, 512)
        self.denseblock5 = self.make_layer(_Dense_Block, 640)
        self.denseblock6 = self.make_layer(_Dense_Block, 768)
        self.bottleneck = nn.Conv2d(in_channels=896, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)

        self.ps = nn.PixelShuffle(self.scale)
        out_dim = int(256/self.scale/self.scale)
        self.reconstruction = nn.Conv2d(in_channels=out_dim, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.Laplacian = Laplacian()

        en = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
              nn.LeakyReLU(0.2)]
        self.en = nn.Sequential(*en)

        self.denseblock_e1 = self.make_layer(_Dense_Block, 64)
        self.denseblock_e2 = self.make_layer(_Dense_Block, 192)
        self.denseblock_e3 = self.make_layer(_Dense_Block, 320)
        self.bottleneck_2 = nn.Conv2d(in_channels=448, out_channels=64 , kernel_size=1, stride=1, padding=0, bias=False)

        self.e8 = nn.Conv2d(in_channels=448, out_channels=256, kernel_size=3, stride=1, padding=1)

        mask = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
              nn.LeakyReLU(0.2),
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
              nn.LeakyReLU(0.2)]
        self.mask = nn.Sequential(*mask)
        self.ps2 = nn.PixelShuffle(self.scale)
        out_dim = int(256 / self.scale / self.scale)
        self.reconstruction_2 = nn.Conv2d(in_channels=out_dim, out_channels=3, kernel_size=3, stride=1, padding=1,
                                        bias=False)
        self.get_g_nopadding = Get_gradient_nopadding()
        self.laplacian = Laplacian()
        #weight = torch.FloatTensor([
        #[[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        #[[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        #[[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]])
        #self.weight = nn.Parameter(data = weight,requires_grad = False).cuda()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()

        bic = F.upsample(x, size=(int(x.shape[2]*self.scale), int(x.shape[3]*self.scale)), mode='bilinear')
        
        out = self.lrelu(self.conv1(x))
        out1 = self.denseblock1(out)
        concat = torch.cat([out,out1], 1)
        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock5(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock6(concat)
        concat = torch.cat([concat,out], 1)
        #out = self.denseblock7(concat)
        #concat = torch.cat([concat,out], 1)
        out = self.bottleneck(concat)
        out = self.ps(out)

        sr_base = self.reconstruction(out) + bic
        
        
        x_fa = self.laplacian(sr_base)
        #pdb.set_trace()

        x_f = self.en(x_fa.cuda())

        x_f2 = self.denseblock_e1(x_f)
        concat = torch.cat([x_f,x_f2], 1)
        x_f = self.denseblock_e2(concat)
        concat = torch.cat([concat,x_f], 1)
        x_f = self.denseblock_e3(concat)
        concat = torch.cat([concat,x_f], 1)
        x_f = self.lrelu(self.e8(concat))

        x_mask = self.mask(self.get_g_nopadding(sr_base))
        frame_mask = torch.sigmoid(x_mask)
        x_frame = frame_mask * x_f +x_f
        #x_frame = self.bottleneck_2(x_frame)
        x_frame = self.ps2(x_frame)
        x_frame = self.reconstruction_2(x_frame)

        x_sr = x_frame + sr_base - x_fa
        frame_e = x_frame - x_fa

        return frame_e, sr_base, x_sr


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


if __name__ == '__main__':
    model = Generator(4).cuda()
    img = torch.rand(3,64,64)
    #img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img=img.cuda()

    out=model(img)

