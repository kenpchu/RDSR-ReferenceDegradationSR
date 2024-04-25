import torch
import torch.nn as nn
from util import make_1ch, make_3ch, resize_tensor_w_kernel
from torch.autograd import Variable
from network.blindsr import make_model as make_dasr_model
from networks import weights_init_G_DN, weights_init_G_UP
import torch.nn.functional as F


def make_dasr_network(conf):
    scale = conf.scale_factor
    conf.scale = [scale]
    dasr_model = make_dasr_model(conf)
    conf.scale = scale
    dasr_model.eval()
    return dasr_model


class CA_WithDR(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_WithDR, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, dr):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        B, C, H, W = x.size()
        b, c = dr.size()
        expand_dr = dr.repeat(1, B // b).view(B, c)
        att = self.conv_du(expand_dr[:, :, None, None])

        return x * att


class Generator_DN_DRCA(nn.Module):
    def __init__(self, dn_network, features=64, reduction=8):
        super(Generator_DN_DRCA, self).__init__()
        self.struct = [7, 5, 3, 1, 1, 1]
        self.DN = dn_network
        # Embedded Module
        self.ca = CA_WithDR(features, features, reduction)

        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, dr):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        compress_dr = self.compress(dr)
        x = make_1ch(x)
        x = self.DN.first_layer(x)
        for i in range(len(self.struct) - 2):
            x = self.DN.feature_block[i](x)
            if i < 2:
                x = self.ca(x, compress_dr)
        out = self.DN.final_layer(x)
        out = make_3ch(out)
        return out


class Generator_DN_DRCA_V2(nn.Module):
    def __init__(self, dn_network, features=64, reduction=8):
        super(Generator_DN_DRCA_V2, self).__init__()
        self.struct = [7, 5, 3, 1, 1, 1]
        self.DN = dn_network
        # Embedded Module
        self.ca = CA_WithDR(features, features, reduction)

        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, dr):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        compress_dr = self.compress(dr)
        x = make_1ch(x)
        x = self.DN.first_layer(x)
        for i in range(len(self.struct) - 2):
            x = self.DN.feature_block[i](x)
            if i == 1:
                x = self.ca(x, compress_dr)
        out = self.DN.final_layer(x)
        out = make_3ch(out)
        return out


class Generator_UP_DRCA(nn.Module):
    def __init__(self, up_network, channels=3, layers=8, features=64, scale_factor=2, reduction=8):
        super(Generator_UP_DRCA, self).__init__()
        self.UP = up_network
        # Embedded Module
        self.ca = CA_WithDR(features, features, reduction)
        self.scale_factor = scale_factor
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, dr):
        x_bilinear = self.UP.bilinear_upsample(x)
        if self.scale_factor == 4:
            x_bilinear = self.UP.bilinear_upsample(x_bilinear)
        ca_layer_indices = [3, 5, 7, 9, 11, 13]
        ca_layer_indices = [13]
        compress_dr = self.compress(dr)
        x = x_bilinear
        for i in range(len(self.UP.model)):
            x = self.UP.model[i](x)
            if i in ca_layer_indices:
                x = self.ca(x, compress_dr)
        # x = self.ca(x, compress_dr)
        out = x_bilinear + x
        return out


class Generator_UP_DRCA_V2(nn.Module):
    def __init__(self, up_network, channels=3, layers=8, features=64, scale_factor=2, reduction=8):
        super(Generator_UP_DRCA_V2, self).__init__()
        self.UP = up_network
        # Embedded Module
        self.ca = CA_WithDR(features, features, reduction)
        self.scale_factor = scale_factor
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, dr):
        x_bilinear = self.UP.bilinear_upsample(x)
        if self.scale_factor == 4:
            x_bilinear = self.UP.bilinear_upsample(x_bilinear)
        # ca_layer_indices = [3, 5, 7, 9, 11, 13]
        # ca_layer_indices = [13]
        # ca_layer_indices = [7, 11]
        ca_layer_indices = [3, 7]

        compress_dr = self.compress(dr)
        x = x_bilinear
        for i in range(len(self.UP.model)):
            x = self.UP.model[i](x)
            if i in ca_layer_indices:
                x = self.ca(x, compress_dr)
        # x = self.ca(x, compress_dr)
        out = x_bilinear + x
        return out

class Generator_UP_DRCA_V3(nn.Module):
    def __init__(self, up_network, channels=3, layers=8, features=64, scale_factor=2, reduction=8):
        super(Generator_UP_DRCA_V3, self).__init__()
        self.UP = up_network
        # Embedded Module
        self.ca = CA_WithDR(features, features, reduction)
        self.scale_factor = scale_factor
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def bicubic_upsample(self, x):
        # bq_up_img = resize_tensor_w_kernel(im_t=x, k=self.bicubic_kernel, sf=(1/self.scale_factor))
        bq_up_img = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        return bq_up_img

    def forward(self, x, dr):
        x_bilinear = self.UP.bilinear_upsample(x)
        if self.scale_factor == 4:
            x_bilinear = self.UP.bilinear_upsample(x_bilinear)

        x_bilinear = self.bicubic_upsample(x)

        ca_layer_indices = [3, 7]
        ca_layer_indices = []

        compress_dr = self.compress(dr)
        x = x_bilinear
        for i in range(len(self.UP.model)):
            x = self.UP.model[i](x)
            if i in ca_layer_indices:
                x = self.ca(x, compress_dr)
        # x = self.ca(x, compress_dr)
        out = x_bilinear + x
        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map, dr):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(dr[:, :, None, None])

        return feature_map * att


class DAC_conv(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, reduction=8):
        super(DAC_conv, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

        self.ca_layer = CA_layer(in_chs, out_chs, reduction)

    def forward(self, in_fea, dr):
        out_feature = self.conv(in_fea)
        out_feature = self.ca_layer(out_feature, dr)
        return out_feature


class Generator_R_UP(nn.Module):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=2):
        super(Generator_R_UP, self).__init__()
        self.scale_factor = scale_factor

        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        self.layers = layers
        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]

        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

        def common_conv(in_chs, out_chs):
            return nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)

        rd_module_head = [common_conv(channels, features), nn.LeakyReLU(0.1, True)]
        self.rd_head = nn.Sequential(*rd_module_head)
        # for i in range(1, layers - 1):
        #     rd_model += [CA_conv(features, features)]

        rd_module_body = [
            CA_conv(features, features) for _ in range(1, layers - 1)
        ]
        self.rd_body = nn.Sequential(*rd_module_body)

        rd_module_tail = [common_conv(features, channels)]
        self.rd_tail = nn.Sequential(*rd_module_tail)

        # self.rd_model = nn.Sequential(*rd_model)

        self.bilinear_kernel = torch.FloatTensor([[[[9/16, 3/16], [3/16, 1/16]]],
                                                  [[[3/16, 9/16], [1/16, 3/16]]],
                                                  [[[3/16, 1/16], [9/16, 3/16]]],
                                                  [[[1/16, 3/16], [3/16, 9/16]]]]).cuda()

        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def bilinear_upsample(self, x):
        x = torch.cat([x[:,:,:1,:], x, x[:,:,-1:,:]], dim=2)
        x = torch.cat([x[:,:,:,:1], x, x[:,:,:,-1:]], dim=3)
        x = make_1ch(x)
        x = F.conv2d(x, self.bilinear_kernel)
        x = F.pixel_shuffle(x, 2)
        x = make_3ch(x)
        x = x[..., 1:-1, 1:-1]
        return x

    def forward(self, x, degra_repre):
        # up_x = self.bilinear_upsample(x)
        # out = up_x + self.model(up_x)
        # return out

        up_x = self.bilinear_upsample(x)
        # out = x + self.model(x)
        # m_out = self.model(x)

        compress_dr = self.compress(degra_repre)
        out = self.rd_head(up_x)
        for i in range(self.layers - 2):
            out = self.rd_body[i](out, compress_dr)
        out = self.rd_tail(out)

        out_final = up_x + out

        return out_final


class Generator_R_DN(nn.Module):

    def __init__(self, represent_fea=None, features=64):
        super(Generator_R_DN, self).__init__()
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        self.represent_fea = represent_fea
        self.layers = len(struct)

        # self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)
        self.first_layer = nn.Conv2d(in_channels=3, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        dn_middle_layer = []
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3:
                dn_middle_layer += [DAC_conv(features, features, struct[layer], 2)]
            else:
                dn_middle_layer += [DAC_conv(features, features, struct[layer], 1)]

        self.rd_dn_layer = nn.Sequential(*dn_middle_layer)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, dr):
        compress_dr = self.compress(dr)
        # x = make_1ch(x)
        out = self.first_layer(x)
        # x = self.feature_block(x)
        for i in range(self.layers - 2):
            out = self.rd_dn_layer[i](out, compress_dr)

        out = self.final_layer(out)
        # return make_3ch(out)
        return out


def weights_init_G_R_DN(m):
    """ initialize weights of the generator """
    weights_init_G_DN(m)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_G_R_UP(m):
    """ initialize weights of the generator """
    weights_init_G_UP(m)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)