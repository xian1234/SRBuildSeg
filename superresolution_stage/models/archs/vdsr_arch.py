from models.archs import common
import torch
import torch.nn.functional as F
  
import torch.nn as nn
import torch.nn.init as init

def make_model(args, parent=False):
    return VDSR(args)

class VDSR(nn.Module):
    def __init__(self, n_resblocks=32, n_feats=256, rgb_range=1, n_colors=3, conv=common.default_conv):
        super(VDSR, self).__init__()

        kernel_size = 3
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.std = torch.Tensor(rgb_std).view(1, 3, 1, 1)

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)

        x = (x - self.mean) / self.std

        res = self.body(x)
        res += x
        x = res * self.std + self.mean
        return x

