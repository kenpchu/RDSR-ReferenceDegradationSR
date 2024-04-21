import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.img_utils import make_1ch, make_3ch, calc_curr_k


def make_downsample_network(conf):
    dn_net = DownSimpleNet(conf).cuda()
    dn_net.apply(weights_init_G_DN)
    cal_kernel = calc_curr_k(dn_net.parameters())
    return dn_net


def make_downsample_x4_network(conf):
    dn_net = DownSimpleNetx4(conf).cuda()
    dn_net.apply(weights_init_G_DN)
    cal_kernel = calc_curr_k(dn_net.parameters())

    return dn_net


def make_dn_x4_k21_network(conf):
    dn_net = DownSampleNetx4K21(conf).cuda()
    dn_net.apply(weights_init_G_DN)
    cal_kernel = calc_curr_k(dn_net.parameters())
    return dn_net


def make_dn_x4_k33_network(conf):
    dn_net = DownSampleNetx4K33(conf).cuda()
    dn_net.apply(weights_init_G_DN)
    cal_kernel = calc_curr_k(dn_net.parameters())
    return dn_net


def make_down_double_network(conf):
    dn_net = DownDoubleNet(conf).cuda()
    dn_net.apply(weights_init_G_DN)

    return dn_net


def make_cycle_dn_network(conf):
    dn_net = DownSimpleNet(conf)
    dn_net.apply(weights_init_G_DN)
    return dn_net


def make_cycle_dn_network(conf):
    dn_net = DownSimpleNet(conf)
    dn_net.apply(weights_init_G_DN)
    return dn_net


def make_cycle_dn_network2(conf):
    dn_net = DownSimpleNet2(conf)
    dn_net.apply(weights_init_G_DN)
    return dn_net


def make_blind_downsample_network(conf):
    return DownSimpleNet(conf)


def make_dn_discriminator_net():
    discriminator_dn = DiscriminatorDN().cuda()
    discriminator_dn.apply(weights_init_D_DN)

    # for i, p in enumerate(discriminator_dn.parameters()):
    #     print(p.name, p.data)
    #     pass

    return discriminator_dn


def weights_init_discriminator(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G_DN(m):
    if m.__class__.__name__.find('Conv') != -1:
        n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
        m.weight.data.normal_(1/n, 1/n)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_D_DN(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DownSimpleNet(nn.Module):
    def __init__(self, conf, features=64):
        super(DownSimpleNet, self).__init__()
        # Reference from Generator_R_Dn
        self.conf = conf
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        self.layers = len(struct)
        # First layer
        if conf.scale == 4:
            self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=2, bias=False)
        else:
            self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        x = make_1ch(x)
        out = self.first_layer(x)
        for i in range(self.layers - 2):
            out = self.feature_block[i](out)

        out = self.final_layer(out)
        return make_3ch(out)


class DownSimpleNet2(nn.Module):
    def __init__(self, conf, features=64):
        super(DownSimpleNet2, self).__init__()
        # Reference from Generator_R_Dn
        self.conf = conf
        struct = [7, 3, 3, 3, 1]
        self.G_kernel_size = 11
        self.layers = len(struct)
        # First layer
        if conf.scale == 4:
            self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=2, bias=False)
        else:
            self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3 and layer == 3:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        x = make_1ch(x)
        out = self.first_layer(x)
        for i in range(self.layers - 2):
            out = self.feature_block[i](out)

        out = self.final_layer(out)
        return make_3ch(out)


class DownSimpleNetx4(nn.Module):
    def __init__(self, conf, features=64):
        super(DownSimpleNetx4, self).__init__()
        # Reference from Generator_R_Dn
        self.conf = conf
        struct = [11, 9, 7, 5, 3, 3, 1]
        # struct = [11, 7, 3, 3, 1]
        # struct = [9, 7, 5, 3, 3, 1]
        # 11, 19, 25, 29, 31
        # 9, 15, 19, 21, 31
        self.G_kernel_size = 33
        self.layers = len(struct)
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)
        # self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False,
        #                              padding=((struct[0]-1)//2, (struct[0]-1)//2), padding_mode='reflect')

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3:
                # feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer],
                #                             stride=2, padding=((struct[layer]-1)//2, (struct[layer]-1)//2),
                #                             padding_mode='reflect', bias=False)]
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer],
                                            stride=2, bias=False)]
            elif struct[layer] == 1:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
            else:
                # feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer],
                #                             padding=((struct[layer]-1)//2, (struct[layer]-1)//2),
                #                             padding_mode='reflect', bias=False)]
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        x = make_1ch(x)
        out = self.first_layer(x)
        for i in range(self.layers - 2):
            out = self.feature_block[i](out)

        out = self.final_layer(out)
        return make_3ch(out)


class DownSampleNetx4K21(DownSimpleNetx4):
    def __init__(self, conf, features=64):
        super(DownSampleNetx4K21, self).__init__(conf)
        struct = [11, 7, 5, 3, 1]
        self.G_kernel_size = 23
        self.layers = len(struct)
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)
        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3 or struct[layer] == 5:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer],
                                            stride=2, bias=False)]
            elif struct[layer] == 1:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)


class DownSampleNetx4K33(DownSimpleNetx4):
    def __init__(self, conf, features=64):
        super(DownSampleNetx4K33, self).__init__(conf)
        struct = [13, 9, 7, 5, 3, 1]
        # 13, 9, 7, 5, 3, 3, 1  = 13 + 8 + 6 + 4 + 1 + 1
        self.G_kernel_size = 33
        self.layers = len(struct)
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)
        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3 or struct[layer] == 5:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer],
                                            stride=2, bias=False)]
            elif struct[layer] == 1:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)


class DownDoubleNet(nn.Module):
    def __init__(self, conf, features=64):
        super(DownDoubleNet, self).__init__()
        # Reference from Generator_R_Dn
        self.conf = conf
        struct = [7, 5, 3, 3, 1, 1]
        self.G_kernel_size = 13
        self.layers = len(struct)
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        x = make_1ch(x)
        out = self.first_layer(x)
        for i in range(self.layers - 2):
            out = self.feature_block[i](out)

        out = self.final_layer(out)
        return make_3ch(out)


class DownResidualNet(nn.Module):
    def __init__(self, conf, features=64):
        super(DownResidualNet, self).__init__()
        # Reference from Generator_R_Dn
        self.conf = conf
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        self.layers = len(struct)
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        x = make_1ch(x)
        out = self.first_layer(x)
        for i in range(self.layers - 2):
            out = self.feature_block[i](out)

        out = self.final_layer(out)
        return make_3ch(out)


class DiscriminatorDN(nn.Module):

    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(DiscriminatorDN, self).__init__()

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

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights.cuda(), bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            return out
            # out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            # return out
        else:
            # x = x.reshape([x.shape[0], 4 * self.channel_in, x.shape[3], x.shape[4]])
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            # out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            # out = torch.transpose(out, 1, 2)
            out = torch.transpose(x, 1, 2)
            # out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[3], x.shape[4]])
            return F.conv_transpose2d(out, self.haar_weights.cuda(), bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac
