import functools
import torch.nn as nn
import torch.nn.functional as F


class KernelAE_base(nn.Module):
    '''Kernel AutoEncoder
    Input kernel size: 21 * 21
    Output kernel size: 32 * 32

    VGG style
    No downsampling
    Linear
    Global Pooling
    No BN
    '''

    def __init__(self, nf=64):
        super(KernelAE_base, self).__init__()

        class Encoder_base(nn.Module):
            def __init__(self, nf=64):
                super(Encoder_base, self).__init__()

                # encoder part (padding 2), receptive field: 5 + 2*4 = 13
                self.conv_0 = nn.Conv2d(1, nf, 5, 1, 2, bias=True)
                self.conv_1 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
                self.conv_2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
                self.conv_3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
                self.conv_4 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
                self.linear = nn.Linear(nf, nf, bias=True)
                self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                fea = self.conv_0(x)
                fea = self.conv_1(fea)
                fea = self.conv_2(fea)
                fea = self.conv_3(fea)
                fea = self.conv_4(fea)
                fea = self.globalpooling(fea).view(fea.size(0), -1)
                fea = self.linear(fea).view(fea.size(0), fea.size(1), 1, 1)
                return fea

        class Encoder_nopad(nn.Module):
            def __init__(self, ngf=64):
                super(Encoder_nopad, self).__init__()
                # encoder part (no padding)
                self.conv_0 = nn.Conv2d(1, nf, 5, 1, 0, bias=False)
                self.conv_1 = nn.Conv2d(nf, nf, 5, 1, 0, bias=False)
                self.conv_2 = nn.Conv2d(nf, nf, 5, 1, 0, bias=False)
                self.conv_3 = nn.Conv2d(nf, nf, 5, 1, 0, bias=False)
                self.conv_4 = nn.Conv2d(nf, nf, 5, 1, 0, bias=False)
                self.linear = nn.Linear(nf, nf, bias=False)

            def forward(self, x):
                fea = self.conv_0(x)
                fea = self.conv_1(fea)
                fea = self.conv_2(fea)
                fea = self.conv_3(fea)
                fea = self.conv_4(fea).view(fea.size(0), -1)
                fea = self.linear(fea).view(fea.size(0), fea.size(1), 1, 1)
                return fea

        class Encoder_nopad_BN(nn.Module):
            def __init__(self, ngf=64):
                super(Encoder_nopad_BN, self).__init__()
                # encoder part (no padding)
                self.conv_0 = nn.Conv2d(1, nf, 5, 1, 0, bias=True)
                self.bn_0 = nn.BatchNorm2d(nf)
                self.conv_1 = nn.Conv2d(nf, nf, 5, 1, 0, bias=True)
                self.bn_1 = nn.BatchNorm2d(nf)
                self.conv_2 = nn.Conv2d(nf, nf, 5, 1, 0, bias=True)
                self.bn_2 = nn.BatchNorm2d(nf)
                self.conv_3 = nn.Conv2d(nf, nf, 5, 1, 0, bias=True)
                self.bn_3 = nn.BatchNorm2d(nf)
                self.conv_4 = nn.Conv2d(nf, nf, 5, 1, 0, bias=True)
                self.bn_4 = nn.BatchNorm2d(nf)
                self.linear = nn.Linear(nf, nf, bias=True)

            def forward(self, x):
                fea = self.bn_0(self.conv_0(x))
                fea = self.bn_1(self.conv_1(fea))
                fea = self.bn_2(self.conv_2(fea))
                fea = self.bn_3(self.conv_3(fea))
                fea = self.bn_4(self.conv_4(fea)).view(fea.size(0), -1)
                fea = self.linear(fea).view(fea.size(0), fea.size(1), 1, 1)
                return fea

        class Decoder(nn.Module):
            '''modified from DCGAN'''

            def __init__(self, nz=64, ngf=64):
                super(Decoder, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.Conv2d(ngf, 1, 5, 1, 2, bias=False),
                    nn.Tanh())

            def forward(self, x):
                return self.main(x)

        class Decoder_noReLU(nn.Module):
            '''modified from DCGAN'''

            def __init__(self, nz=64, ngf=64):
                super(Decoder_noReLU, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    # state size. (ngf) x 32 x 32
                    nn.Conv2d(ngf, 1, 5, 1, 2, bias=False))

            def forward(self, x):
                return self.main(x)

        class Decoder_noBNReLU(nn.Module):
            '''modified from DCGAN'''

            def __init__(self, nz=64, ngf=64):
                super(Decoder_noBNReLU, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    # state size. (ngf) x 32 x 32
                    nn.Conv2d(ngf, 1, 5, 1, 2, bias=False))

            def forward(self, x):
                return self.main(x)

        self.encoder = Encoder_nopad(nf)
        self.decoder = Decoder_noBNReLU(nf, nf)

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return code, out

    def forward_code(self, code):
        # code = self.encoder(x)
        out = self.decoder(code)
        return out


if __name__ == "__main__":
    import torch
    inp = torch.ones(1, 1, 21, 21).float().cuda()
    net = KernelAE_base().cuda()
    print(net)
    code, out = net(inp)
    print(code.size(), out.size())
