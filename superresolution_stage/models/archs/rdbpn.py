import os
import torch.nn as nn
import torch.optim as optim
from models.archs.base_networks import *
from torchvision.transforms import *


class RDBPNblock(torch.nn.Module):
    def __init__(self, num_filter, kernel=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(RDBPNblock, self).__init__()

        self.mergeh1 = nn.Conv2d(2*num_filter, num_filter, 1, 1, 0, bias=bias)
        self.mergel1 = nn.Conv2d(2*num_filter, num_filter, 1, 1, 0, bias=bias)
        
        self.mergeh2 = nn.Conv2d(3*num_filter, num_filter, 1, 1, 0, bias=bias)
        self.mergel2 = nn.Conv2d(3*num_filter, num_filter, 1, 1, 0, bias=bias)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
            
        self.up1 = UpBlock(num_filter, kernel, stride, padding)
        self.down1 = DownBlock(num_filter, kernel, stride, padding)
        self.up2 = UpBlock(num_filter, kernel, stride, padding)
        self.down2 = DownBlock(num_filter, kernel, stride, padding)
        self.up3 = UpBlock(num_filter, kernel, stride, padding)
        self.down3 = DownBlock(num_filter, kernel, stride, padding)
        
        

    def forward(self, x):
    
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        merge_h = self.mergeh1(concat_h)
        
        l = self.down2(merge_h)
        
        concat_l = torch.cat((l, l1),1)
        merge_l = self.mergel1(concat_l)
        
        h = self.up3(merge_l)
        concat_h = torch.cat((h, concat_h),1)
        merge_h = self.mergeh2(concat_h)
        
        l = self.down3(merge_h)
        
        concat_l = torch.cat((l, concat_l),1)
        merge_l = self.mergel2(concat_l)

        return torch.add(0.2*merge_l,x)


class RDBPN(nn.Module):
    def __init__(self, num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=4):
        super(RDBPN, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        

        #Back-projection stages
        self.up10 = UpBlock(base_filter, kernel, stride, padding)
        
        self.rdbpn1 = RDBPNblock(base_filter)
        self.rdbpn2 = RDBPNblock(base_filter)
        self.rdbpn3 = RDBPNblock(base_filter)
        #Reconstruction
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        x = self.feat0(x)
        
        res = self.rdbpn1(x)
        res = self.rdbpn2(res)
        res = self.rdbpn3(res)
        
        res = self.conv(res)      
        x = torch.add(res,x)
        
        h = self.up10(x)     
        x = self.output_conv(h)
        
        return x

