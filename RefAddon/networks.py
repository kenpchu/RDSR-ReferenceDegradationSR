import torch
import torch.nn as nn
import numpy as np
from util import make_1ch, make_3ch, calc_curr_k, resize_tensor_w_kernel, analytic_kernel_w
import torch.nn.functional as F


class Generator_UP(nn.Module):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=2):
        super(Generator_UP, self).__init__()
        self.scale_factor = scale_factor
        
        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(True)]
        
        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(True)]
        
        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]  
        
        self.model = nn.Sequential(*model)
        
        self.bilinear_kernel = torch.FloatTensor([[[[9/16, 3/16], [3/16, 1/16]]],
                                                  [[[3/16, 9/16], [1/16, 3/16]]],
                                                  [[[3/16, 1/16], [9/16, 3/16]]],
                                                  [[[1/16, 3/16], [3/16, 9/16]]]]).cuda()
    
    def bilinear_upsample(self, x):
        x = torch.cat([x[:,:,:1,:], x, x[:,:,-1:,:]], dim=2)
        x = torch.cat([x[:,:,:,:1], x, x[:,:,:,-1:]], dim=3)        
        x = make_1ch(x)
        x = F.conv2d(x, self.bilinear_kernel)
        x = F.pixel_shuffle(x, 2)
        # x = F.pixel_shuffle(x, 2.5)
        x = make_3ch(x)
        x = x[..., 1:-1, 1:-1]
        return x
        
    def forward(self, x):
        x = self.bilinear_upsample(x)
        out = x + self.model(x)  # add skip connections
        return out


class Generator_UP_V2(nn.Module):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=2, group=2):
        super(Generator_UP_V2, self).__init__()
        self.scale_factor = scale_factor

        model_head = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        self.head = nn.Sequential(*model_head)

        # modules_body = [
        #     DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
        #     for _ in range(self.n_groups)
        # ]
        # modules_body.append(conv(n_feats, n_feats, kernel_size))
        # self.body = nn.Sequential(*modules_body)

        # for i in range(layers):
        #     model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
        #               nn.ReLU(True)]
        self.group = group
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

        modules_body = []
        for _ in range(layers):
            modules_body += [
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True)
            ]
        self.residual_body = nn.Sequential(*modules_body)

        model_tail = [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]
        self.tail = nn.Sequential(*model_tail)

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
        # x = F.pixel_shuffle(x, 2.5)
        x = make_3ch(x)
        x = x[..., 1:-1, 1:-1]
        return x

    def forward(self, x):
        x_bilinear = self.bilinear_upsample(x)
        out = self.head(x_bilinear)
        for _ in range(self.group):
            res = out
            res = self.residual_body(res)
            res = self.conv(res)
            res += out
            out = res

        final_out = x_bilinear + self.tail(out)
        return final_out


class Generator_UP_x4(Generator_UP):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=4):
        super(Generator_UP_x4, self).__init__(channels=3, layers=8, features=64, scale_factor=4)

    def forward(self, x):
        x = self.bilinear_upsample(x)
        x = self.bilinear_upsample(x)
        out = x + self.model(x)  # add skip connections
        return out


class Generator_UP_x4_V2(Generator_UP):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=4):
        super(Generator_UP_x4_V2, self).__init__(channels=3, layers=8, features=64, scale_factor=4)

    def bicubic_upsample(self, x):
        # bq_up_img = resize_tensor_w_kernel(im_t=x, k=self.bicubic_kernel, sf=(1/self.scale_factor))
        bq_up_img = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        return bq_up_img

    def forward(self, x):
        # x = self.bilinear_upsample(x)
        # x = self.bilinear_upsample(x)
        x = self.bicubic_upsample(x)


        out = x + self.model(x)  # add skip connections
        return out


class Generator_DN(nn.Module):
    def __init__(self, features=64):
        super(Generator_DN, self).__init__()
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3: # Downsample on the first layer with kernel_size=1
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        x = make_1ch(x)
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        out = make_3ch(out)
        return out


class Generator_DN_V2(nn.Module):
    def __init__(self, features=64):
        super(Generator_DN_V2, self).__init__()

        # struct = [7, 5, 3, 1, 1, 1]
        # self.G_kernel_size = 13
        struct = [7, 3, 3, 3, 1, 1]
        self.G_kernel_size = 13
        # struct = [7, 5, 3, 3, 1, 1]
        # self.G_kernel_size = 15

        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            # if struct[layer] == 3 and layer == 3: # Downsample on the first layer with kernel_size=1
            if struct[layer] == 3 and layer == 3:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        x = make_1ch(x)
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return make_3ch(out)


class Generator_DN_x4(nn.Module):
    def __init__(self, features=64):
        super(Generator_DN_x4, self).__init__()
        struct = [7, 5, 3, 1, 1, 1]
        # self.G_kernel_size = 13
        self.G_kernel_size = 13 * 3 - 2 - (13 // 2) * 2

        # First layer
        # self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=2, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3: # Downsample on the first layer with kernel_size=1
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        x = make_1ch(x)
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        out = make_3ch(out)
        return out

    def calc_ker(self):
        curr_k = calc_curr_k(self.parameters())
        curr_k_x4 = analytic_kernel_w(curr_k.cuda())
        return curr_k_x4


class Generator_DN_x4_Kernel(nn.Module):
    def __init__(self, base_dn):
        super(Generator_DN_x4_Kernel, self).__init__()
        self.base_DN = base_dn
        self.scale_factor = 0.25
        self.G_kernel_size = 13 * 3 - 2 - (13//2) * 2

    def forward(self, x):
        curr_k_x4 = self.calc_ker()
        x_dn_x4 = resize_tensor_w_kernel(im_t=x, k=curr_k_x4, sf=self.scale_factor)
        return x_dn_x4

    def calc_ker(self):
        curr_k = calc_curr_k(self.base_DN.parameters())
        curr_k_x4 = analytic_kernel_w(curr_k.cuda())
        return curr_k_x4


class Discriminator_DN(nn.Module):

    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(Discriminator_DN, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=features, kernel_size=D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(features),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())
        
        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = 128 - self.forward(torch.FloatTensor(torch.ones([1, 3, 128, 128]))).shape[-1]
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return out


class Discriminator_UP(nn.Module):
    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(Discriminator_UP, self).__init__()

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


def weights_init_G_DN(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
        m.weight.data.normal_(1/n, 1/n)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def weights_init_G_UP(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


class DownScale(object):
    def __init__(self, kernel, stride=2):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, hr_tensor):
        hr_img = make_1ch(hr_tensor)
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        lr_img = F.conv2d(hr_img, kernel, stride=self.stride, padding=0)
        lr_img = make_3ch(lr_img)
        return lr_img


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))


class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0), torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1), torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


class DegradationProcessing(object):
    def __init__(self,
                 conf,
                 kernel,
                 mode='bicubic'):

        self.scale = conf.scale_factor
        self.mode = mode
        self.noise = conf.noise
        self.kernel = kernel
        self.blur = BatchBlur(kernel_size=kernel.size()[-1])
        self.bicubic = Bicubic()

    def __call__(self, hr_tensor):
        with torch.no_grad():

            B, C, H, W = hr_tensor.size()

            # blur
            hr_blured = self.blur(hr_tensor.view(B, -1, H, W), self.kernel)
            hr_blured = hr_blured.view(-1, C, H, W)  # BN, C, H, W

            # downsampling
            if self.mode == 'bicubic':
                lr_blured = self.bicubic(hr_blured, scale=1/self.scale)
            elif self.mode == 's-fold':
                lr_blured = hr_blured.view(-1, C, H//self.scale, self.scale, W//self.scale, self.scale)[:, :, :, 0, :, 0]

            # add noise
            if self.noise > 0:
                _, C, H_lr, W_lr = lr_blured.size()
                noise_level = torch.rand(B, 1, 1, 1).to(lr_blured.device) * self.noise if random else self.noise
                noise = torch.randn_like(lr_blured).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
                lr_blured.add_(noise)

            lr_blured = torch.clamp(lr_blured.round(), 0, 255)

            return lr_blured.view(B, C, H//int(self.scale), W//int(self.scale))
