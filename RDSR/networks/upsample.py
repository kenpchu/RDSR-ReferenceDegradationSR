import torch.nn as nn
import torch
import functools
import torch.nn.functional as F

from utils.img_utils import make_1ch, make_3ch

from networks.downsample import make_downsample_network
from networks.blindsr import make_model as make_dasr_model
from networks.module_util import make_layer, ResidualBlockNoBN, initialize_weights


def make_upsample_network(conf):
    sr_net = SimpleUpSample(conf).cuda()
    sr_net.apply(weights_init_upsample)
    return sr_net


def make_cycle_sr_network(conf):
    sr_net = SimpleUpSample(conf)
    sr_net.apply(weights_init_upsample)
    return sr_net


def make_upsample_residual_network(conf):
    return ResidualUpSample(conf)


def make_blind_upsample_residual_network(conf):
    return SimpleUpSample(conf)


def make_dasr_network(conf):
    scale = conf.scale
    conf.scale = [scale]
    dasr_model = make_dasr_model(conf)
    conf.scale = scale
    dasr_model.eval()
    return dasr_model


def make_up_discriminator_net(device):
    discriminator_up = DiscriminatorUP().to(device)
    discriminator_up.apply(weights_init_disc_up)

    return discriminator_up


class SimpleUpSample(nn.Module):
    def __init__(self, conf=None, channels=3, layers=8, features=64, scale_factor=2):
        super(SimpleUpSample, self).__init__()

        self.scale_factor = conf.scale

        # model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        #          nn.ReLU(True)]
        #
        # for i in range(1, layers - 1):
        #     model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        #               nn.ReLU(True)]
        #
        # model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')]

        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(True)]

        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(True)]

        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

        self.bilinear_kernel = torch.FloatTensor([[[[9 / 16, 3 / 16], [3 / 16, 1 / 16]]],
                                                  [[[3 / 16, 9 / 16], [1 / 16, 3 / 16]]],
                                                  [[[3 / 16, 1 / 16], [9 / 16, 3 / 16]]],
                                                  [[[1 / 16, 3 / 16], [3 / 16, 9 / 16]]]]).cuda()

    def bilinear_upsample(self, x):
        x = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
        x = torch.cat([x[:, :, :, :1], x, x[:, :, :, -1:]], dim=3)
        x = make_1ch(x)
        x = F.conv2d(x, self.bilinear_kernel)
        x = F.pixel_shuffle(x, 2)
        x = make_3ch(x)
        x = x[..., 1:-1, 1:-1]
        return x

    def forward(self, x):
        x = self.bilinear_upsample(x)
        out = x + self.model(x)  # add skip connections
        return out


def weights_init_upsample(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_disc_up(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResidualUpSample(nn.Module):
    def __init__(self, conf=None, channels=3, layers=1, features=64):
        super(ResidualUpSample, self).__init__()
        self.scale = conf.scale

        self.conv_first = nn.Conv2d(channels, features, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlockNoBN, nf=features)
        self.recon_trunk = make_layer(basic_block, layers)

        # if self.scale == 2:
        #     self.upconv1 = nn.Conv2d(features, features * 4, 3, 1, 1, bias=True)
        #     self.pixel_shuffle = nn.PixelShuffle(2)
        # elif self.scale == 3:
        #     self.upconv1 = nn.Conv2d(features, features * 9, 3, 1, 1, bias=True)
        #     self.pixel_shuffle = nn.PixelShuffle(3)
        # elif self.scale == 4:
        #     self.upconv1 = nn.Conv2d(features, features * 4, 3, 1, 1, bias=True)
        #     self.upconv2 = nn.Conv2d(features, features * 4, 3, 1, 1, bias=True)
        #     self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(features, channels, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights(
            [self.conv_first, self.HRconv, self.conv_last], 0.1
        )
        # if self.scale == 4:
        #     initialize_weights(self.upconv2, 0.1)

        self.relu = nn.ReLU(True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
                 self.relu]

        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                      self.relu]

        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def get_base_img(self, x):
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        return base

    def forward(self, x):
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )

        fea = self.lrelu(self.conv_first(base))
        out = self.recon_trunk(fea)
        out = self.conv_last(self.lrelu(self.HRconv(out)))

        # out = self.model(base)

        out += base
        return out


class DiscriminatorUP(nn.Module):

    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(DiscriminatorUP, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=3, out_channels=features, kernel_size=D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, layers - 1):
            feature_block += [nn.utils.spectral_norm(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(features),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1, bias=True)),
            nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = 128 - self.forward(torch.FloatTensor(torch.ones([1, 3, 128, 128]))).shape[-1]

    def forward(self, x):
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return out
