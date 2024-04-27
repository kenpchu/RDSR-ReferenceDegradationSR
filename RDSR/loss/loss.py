import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.img_utils import edge_detect, make_1ch, make_3ch
from utils.sobel import Sobel


class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        # DualSR use L2
        self.loss = nn.MSELoss()
        # kernelGAN use L1
        self.lossl1 = nn.L1Loss(reduction='mean')
        # real and fake labels
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, d_last_layer, is_d_input_real):
        target_tensor = self.get_target_tensor(d_last_layer, is_d_input_real)
        loss = self.loss(d_last_layer, target_tensor)
        return loss


class GANLoss2(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self):
        super(GANLoss2, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        # DualSR use L2
        # self.loss = nn.MSELoss()
        # kernelGAN use L1
        self.loss = nn.L1Loss(reduction='mean')
        # real and fake labels
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, d_last_layer, is_d_input_real):
        target_tensor = self.get_target_tensor(d_last_layer, is_d_input_real)
        loss = self.loss(d_last_layer, target_tensor)
        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class InterpolationLoss(nn.Module):
    def __init__(self, scale):
        super(InterpolationLoss, self).__init__()
        self.scale = scale

    def forward(self, lr, sr):
        mask, lr_bic = edge_detect(lr, self.scale)
        loss = nn.L1Loss(reduction='none')(lr_bic, sr)
        return torch.mean((1-mask)*loss)


class InterpolationLoss2(nn.Module):
    def __init__(self, scale):
        super(InterpolationLoss2, self).__init__()
        self.scale = scale

    def forward(self, lr_bq, sr):
        # lr_bq = imresize(im=lr, scale_factor=self.scale, kernel='cubic')
        sobel_A = Sobel()(lr_bq.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        loss = nn.L1Loss()(lr_bq * loss_map_A, sr * loss_map_A)
        return loss


class HighFrequencyLoss(nn.Module):
    def __init__(self):
        super(HighFrequencyLoss, self).__init__()
        self.laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3).cuda()
        self.gaussian_filter = torch.Tensor([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]).view(1, 1, 3, 3).cuda()

    def forward(self, hr, sr):
        hr = make_1ch(hr)
        sr = make_1ch(sr)
        hr_la = F.conv2d(hr, self.laplacian_filter)
        sr_la = F.conv2d(sr, self.laplacian_filter)
        hr_ga = F.conv2d(hr, self.gaussian_filter)
        sr_ga = F.conv2d(sr, self.gaussian_filter)

        hr_la = make_3ch(hr_la)
        sr_la = make_3ch(sr_la)
        hr_ga = make_3ch(hr_ga)
        sr_ga = make_3ch(sr_ga)

        loss_la = nn.L1Loss(reduction='none')(hr_la, sr_la)
        loss_ga = nn.L1Loss(reduction='none')(hr_ga, sr_ga)

        r_loss = torch.mean(loss_la[:, 0, ...]) + torch.mean(loss_ga[:, 0, ...])
        g_loss = torch.mean(loss_la[:, 1, ...]) + torch.mean(loss_ga[:, 1, ...])
        b_loss = torch.mean(loss_la[:, 2, ...]) + torch.mean(loss_ga[:, 2, ...])

        loss = r_loss + g_loss + b_loss

        return loss


class HighFrequencyLoss2(HighFrequencyLoss):
    def __init__(self):
        super(HighFrequencyLoss2, self).__init__()

    def forward(self, hr, sr):
        sr_la_r = F.conv2d(sr[:,0,...].unsqueeze(1), self.laplacian_filter)
        sr_la_g = F.conv2d(sr[:,1,...].unsqueeze(1), self.laplacian_filter)
        sr_la_b = F.conv2d(sr[:,2,...].unsqueeze(1), self.laplacian_filter)
        sr_ga_r = F.conv2d(sr[:,0,...].unsqueeze(1), self.gaussian_filter)
        sr_ga_g = F.conv2d(sr[:,1,...].unsqueeze(1), self.gaussian_filter)
        sr_ga_b = F.conv2d(sr[:,2,...].unsqueeze(1), self.gaussian_filter)

        hr_la_r = F.conv2d(hr[:, 0, ...].unsqueeze(1), self.laplacian_filter)
        hr_la_g = F.conv2d(hr[:, 1, ...].unsqueeze(1), self.laplacian_filter)
        hr_la_b = F.conv2d(hr[:, 2, ...].unsqueeze(1), self.laplacian_filter)
        hr_ga_r = F.conv2d(hr[:, 0, ...].unsqueeze(1), self.gaussian_filter)
        hr_ga_g = F.conv2d(hr[:, 1, ...].unsqueeze(1), self.gaussian_filter)
        hr_ga_b = F.conv2d(hr[:, 2, ...].unsqueeze(1), self.gaussian_filter)

        loss_r = F.l1_loss(sr_la_r, hr_la_r) + F.l1_loss(sr_ga_r, hr_ga_r)
        loss_g = F.l1_loss(sr_la_g, hr_la_g) + F.l1_loss(sr_ga_g, hr_ga_g)
        loss_b = F.l1_loss(sr_la_b, hr_la_b) + F.l1_loss(sr_ga_b, hr_ga_b)

        loss = loss_r + loss_g + loss_b

        return loss


# TODO: wait for verify
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, hr, sr):
        diff = torch.add(hr, -sr)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
        pass


class CharbonnierLossV2(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLossV2, self).__init__()
        self.eps = eps

    def forward(self, hr, sr):
        hr = hr / 255.
        sr = sr / 255.
        diff = torch.add(hr, -sr)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
        pass


class ColorLoss(nn.Module):
    def __init__(self, scale):
        super(ColorLoss, self).__init__()
        self.scale = scale

    def forward(self, lr, sr):
        # TODO: 0629 start from here
        up_lr = F.interpolate(lr, scale_factor=self.scale, mode='bilinear', align_corners=True)
        loss = F.l1_loss(up_lr, sr)
        return loss


class GradientVariance(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
    """

    def __init__(self, patch_size, cpu=False):
        super(GradientVariance, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = 0.2989 * output[:, 0:1, :, :] + 0.5870 * output[:, 1:2, :, :] + 0.1140 * output[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)

        return gradvar_loss


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

# TODO: SSIM Loss
# TODO: Gradient weighted loss
# video dataset:Vimeo-90k    http://toflow.csail.mit.edu/

# from chatgpt
# def gradient_loss(input_image, target_image):
#     # Compute gradients of input and target images
#     input_gradients = compute_gradients(input_image)
#     target_gradients = compute_gradients(target_image)
#
#     # Compute the mean squared difference between gradients
#     loss = F.mse_loss(input_gradients, target_gradients)
#
#     return loss
#
# def compute_gradients(image):
#     # Convert image to grayscale if necessary
#     if image.size(1) > 1:
#         image = torch.mean(image, dim=1, keepdim=True)
#
#     # Compute horizontal and vertical gradients
#     gradients_x = image[:, :, :, :-1] - image[:, :, :, 1:]
#     gradients_y = image[:, :, :-1, :] - image[:, :, 1:, :]
#
#     # Stack gradients along the channel dimension
#     gradients = torch.cat([gradients_x, gradients_y], dim=1)
#
#     return gradients
