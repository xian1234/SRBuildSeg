from models.archs import common

import torch.nn as nn
import torch

class MDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, n_colors=3,scale=[4], conv=common.default_conv):
        super(MDSR, self).__init__()
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        m_head = [conv(n_colors, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            common.Upsampler(
                conv, s, n_feats, act=False
            ) for s in scale
        ])

        m_tail = [conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)

        return x




