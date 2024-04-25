import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models.vgg import vgg16
from util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, make_1ch, make_3ch


class GANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.MSELoss()
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
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class DownsamplerRegularization(nn.Module):
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0
    
    def __init__(self, scale_factor, G_kernel_size):
        super(DownsamplerRegularization, self).__init__()
        self.criterion_bicubic = DownScaleLoss(scale_factor=scale_factor).cuda()
        self.criterion_sum2one = SumOfWeightsLoss().cuda()
        self.criterion_boundaries = BoundariesLoss(k_size=G_kernel_size).cuda()
        self.criterion_centralized = CentralizedLoss(G_kernel_size, scale_factor=scale_factor).cuda()
        self.criterion_sparse = SparsityLoss().cuda()
        
    def forward(self, curr_k, g_input, g_pred):
        self.loss_bicubic = self.criterion_bicubic.forward(g_input=g_input, g_output=g_pred)
        self.loss_boundaries = self.criterion_boundaries.forward(kernel=curr_k)
        self.loss_sum2one = self.criterion_sum2one.forward(kernel=curr_k)
        self.loss_centralized = self.criterion_centralized.forward(kernel=curr_k)
        self.loss_sparse = self.criterion_sparse.forward(kernel=curr_k)
        
        reg_term = self.loss_bicubic * self.lambda_bicubic + \
                   self.loss_sum2one * self.lambda_sum2one + \
                   self.loss_boundaries * self.lambda_boundaries + \
                   self.loss_centralized * self.lambda_centralized + \
                   self.loss_sparse * self.lambda_sparse
        return reg_term
        
class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
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

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.tensor(1.0).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))).squeeze(), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = torch.FloatTensor(create_penalty_mask(k_size, 30)).cuda()
        self.zero_label = Variable(torch.zeros_like(self.mask).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))


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


class HighFrequencyYLoss(nn.Module):
    def __init__(self):
        super(HighFrequencyYLoss, self).__init__()
        self.laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3).cuda()
        self.gaussian_filter = torch.Tensor([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]).view(1, 1, 3, 3).cuda()

    def img2y(self, rgb_image):
        R = rgb_image[:, 0, ...].unsqueeze(1)
        G = rgb_image[:, 1, ...].unsqueeze(1)
        B = rgb_image[:, 2, ...].unsqueeze(1)
        Y = 0.299 * R + 0.587 * G + 0.114 * B

        return Y

    def forward(self, hr, sr):
        hr_y = self.img2y(hr)
        sr_y = self.img2y(sr)
        hr_la = F.conv2d(hr_y, self.laplacian_filter)
        sr_la = F.conv2d(sr_y, self.laplacian_filter)
        hr_ga = F.conv2d(hr_y, self.gaussian_filter)
        sr_ga = F.conv2d(sr_y, self.gaussian_filter)

        loss_la = nn.L1Loss(reduction='none')(hr_la, sr_la)
        loss_ga = nn.L1Loss(reduction='none')(hr_ga, sr_ga)

        loss = torch.mean(loss_la) + torch.mean(loss_ga)

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


class CoBi_Loss(nn.Module):
    def __init__(self):
        super(CoBi_Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        # vgg.load_state_dict(torch.load('/data/mdyao/city100_calibrate/checkpoint/pretrain/vgg16/vgg16-397923af.pth'))

        # vgg_features = models.vgg16(pretrained=True).features
        # modules = [m for m in vgg_features]
        # # loss_network = nn.Sequential(*modules[:8]).eval()
        # self.vgg = nn.Sequential(*modules[:31]).eval()
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:8])
        for param in self.vgg.parameters():
            param.requires_grad = False


        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False

    def compute_cosine_distance(self, x, y):
        assert x.size() == y.size()
        N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

        y_mu = y.mean(3).mean(2).reshape(N, -1, 1, 1)
        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)
        d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
        return d

    def compute_relative_distance(self, d):
        # ?????min(cosine_sim)
        d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)
        d_tilde = d / (d_min + 1e-5)

        return d_tilde

    def compute_cx(self, d_tilde, h=0.5):
        w = torch.exp((1 - d_tilde) / h)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
        return cx_ij

    def compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

        return feature_grid
    def compute_spatial_loss(self,x,h=0.5):
        # spatial loss
        grid = self.compute_meshgrid(x.shape)
        d = self.compute_cosine_distance(grid, grid)
        d_tilde = self.compute_relative_distance(d)
        cx_sp = self.compute_cx(d_tilde,h)
        return cx_sp

    def compute_feat_loss(self,x,y,h=0.5):
        # feature loss
        d = self.compute_cosine_distance(x,y)
        d_tilde = self.compute_relative_distance(d)
        cx_feat = self.compute_cx(d_tilde,h)
        return cx_feat

    def cobi_vgg(self, out_images, target_images, w=0.1):
        sp_loss = self.compute_spatial_loss(self.vgg(out_images), h=0.5).cuda()
        feat_loss = self.compute_feat_loss(self.vgg(out_images), self.vgg(target_images), h=0.5)
        combine_loss = (1 - w) * feat_loss + w * sp_loss
        return combine_loss

    def cobi_rgb(self,out_images,target_images,w=0.1):
        sp_loss = self.compute_spatial_loss(out_images, h=0.5)
        feat_loss = self.compute_feat_loss(out_images, target_images, h=0.5)
        combine_loss = (1 - w) * feat_loss + w * sp_loss
        return combine_loss

    def forward(self, out_images, target_images, w=0.1):
        loss = self.cobi_vgg(out_images, target_images) + self.cobi_rgb(out_images, target_images)
        return loss