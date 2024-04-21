import random
import numpy as np
import math
import os
import cv2 as cv
import torch

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import filters, measurements, interpolation
from scipy.signal import convolve2d
from scipy.io import loadmat
from scipy import stats


def edge_detect(img, scale):
    upsampler = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=True)
    img_up = upsampler(img)
    img_np = np.sum(tensor2im(img_up), axis=2)

    imx = np.zeros(img_np.shape)
    filters.sobel(img_np, 1, imx)

    imy = np.zeros(img_np.shape)
    filters.sobel(img_np, 0, imy)

    mag = np.hypot(imx, imy)
    mag -= np.min(mag)
    mag /= np.max(mag)
    mag = np.expand_dims(mag, 0)
    mag = np.repeat(mag, 3, axis=0)

    return torch.FloatTensor(mag).unsqueeze(0).cuda(), img_up


def make_1ch(im):
    s = im.shape
    assert s[1] == 3
    return im.reshape(s[0]*3, 1, s[2], s[3])


def make_3ch(im):
    s = im.shape
    assert s[1] == 1
    return im.reshape(s[0] // 3, 3, s[2], s[3])


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t, normalize_en=False, rgb_range=255.0):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    # im_np = np.transpose(move2cpu(im_t)[0], (1, 2, 0))
    im_np = np.clip(np.round(np.transpose(move2cpu(im_t)[0], (1, 2, 0))), 0, 255)
    if normalize_en:
        im_np = (im_np + 1) / 2.0

    im_np = im_np * 255.0 / rgb_range
    im_np = np.clip(im_np, 0, 255)

    return im_np.astype(np.uint8)


def tensor2im2(im_t, normalize_en=False):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.transpose(move2cpu(im_t[0]), (1, 2, 0))
    if normalize_en:
        im_np = (im_np + 1.0) / 2.0
    im_np = np.clip(np.round(im_np * 255.0), 0, 255)

    return im_np.astype(np.uint8)


def tensor_to_255(im_t):
    """Copy the tensor convert to range [0,255]"""
    return im_t * 255


# def tensor2image
def tensor_norm(im_t, normalize_en=True, rgb_range=1.0):
    im_t = im_t * (rgb_range / 255.0)
    if normalize_en:
        im_t = im_t * 2.0 - 1.0
    return im_t


def np2tensor(im_np, normalize_en=False, rgb_range=255.0):
    # Copy the image to the gpu & converts to range [-1,1]
    np_transpose = np.ascontiguousarray(im_np.transpose((2, 0, 1)))
    img_np = np_transpose * (rgb_range / 255.0)
    if normalize_en:
        img_np = img_np * 2.0 - 1.0
    im_t = torch.from_numpy(img_np).float().cuda()

    return im_t


def read_image(path):
    # print(path)
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im


def rgb2gray(im):
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def pad_edges(im, edge):
    zero_padded = np.zeros_like(im)
    zero_padded[edge: -edge, edge: -edge] = im[edge: -edge, edge: -edge]
    return zero_padded


def clip_extreme(im, percent):
    im_sorted = np.sort(im.flatten())
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    return np.clip(im, v_min, v_max) - v_min


def create_gradient_map(im, window=5, percent=0.97):
    gx, gy = np.gradient(rgb2gray(im))
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y , lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()

    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)

    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop//2) ** 2)
    prob_map = pad_edges(blurred, crop // 2)
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten() / prob_map.flatten().shape[0])

    return prob_vec


def nn_interpolation(im, sf):
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def shave_a2b(a, b):
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1

    assert (a.shape[r] >= b.shape[r]) and (a.shape[c] >= b.shape[c])
    assert ((a.shape[r] - b.shape[r]) % 2 == 0) and ((a.shape[c] - b.shape[c]) % 2 == 0)

    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r//2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def tensor2tb_log(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round(move2cpu(im_t)), 0, 255)
    return im_np.astype(np.uint8)


def calculate_psnr(img1, img2):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_psnr2(img1, img2, border=0):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = tensor2im2(img1)
    img2 = tensor2im2(img2)

    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_psnr3(img1, img2):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = move2cpu(img1)
    img2 = move2cpu(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_psnr_y(img1, img2, border=0):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = tensor2im2(img1)
    img2 = tensor2im2(img2)

    img1_y = np.dot(img1 / 255., [65.481, 128.553, 24.966]) + 16
    img2_y = np.dot(img2 / 255., [65.481, 128.553, 24.966]) + 16

    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 10 * np.log10(255 * 255 / mse)


def cal_y_psnr(A, B, border):
    A = A.astype('float64')
    B = B.astype('float64')

    if len(A.shape) == 3:
        # calculate Y channel like matlab 'rgb2ycbcr' function
        Y_A = np.dot(A / 255., [65.481, 128.553, 24.966]) + 16
        Y_B = np.dot(B / 255., [65.481, 128.553, 24.966]) + 16
    else:
        Y_A = A
        Y_B = B

    if border > 0:
        Y_A = Y_A[border:-border, border:-border]
        Y_B = Y_B[border:-border, border:-border]

    e = Y_A - Y_B;
    mse = np.mean(e ** 2);
    psnr_cur = 10 * np.log10(255 * 255 / mse);

    return psnr_cur


def calculate_psnr_np(img1, img2):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = np.clip(img1, 0, 255)
    img1.astype(np.uint8)
    img2 = np.clip(img2, 0, 255)
    img2.astype(np.uint8)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# this is for reference selection
def calculate_rgb_mean(img_path):
    im = read_image(img_path)
    r_mean = np.mean(im[0])
    g_mean = np.mean(im[1])
    b_mean = np.mean(im[2])
    return r_mean, g_mean, b_mean


def calc_rgb_hist(img_path):
    im = read_image(img_path)
    r_hist = cv.calcHist([im], [0], None, [256], [0, 255]) / (im.shape[0] * im.shape[1])
    g_hist = cv.calcHist([im], [1], None, [256], [0, 255]) / (im.shape[0] * im.shape[1])
    b_hist = cv.calcHist([im], [2], None, [256], [0, 255]) / (im.shape[0] * im.shape[1])

    return r_hist, g_hist, b_hist


# this is for reference selection
def sample_ref_by_color_space(tar_img, ref_list, ref_cnt=3):
    tar_rgb = calculate_rgb_mean(tar_img)
    ref_rgb_list = []
    tar_ref_score_list = []
    for idx, ref in enumerate(ref_list):
        ref_rgb = calculate_rgb_mean(ref)
        ref_rgb_list.append(ref_rgb)
        mse_ch = sum([(tar - ref) ** 2 for tar, ref in zip(tar_rgb, ref_rgb)])
        tar_ref_score_list.append((idx, mse_ch))

    sorted_score_list = sorted(tar_ref_score_list, key=lambda x: x[1])
    ref_idx_list = [ind for ind, val in sorted_score_list[:ref_cnt]]

    return ref_idx_list


def sample_ref_by_rgb_histogram(tar_img, ref_list, ref_cnt=3):
    tar_hist = calc_rgb_hist(tar_img)
    ref_hist_list = []
    tar_ref_score_list = []
    for idx, ref in enumerate(ref_list):
        ref_hist = calc_rgb_hist(ref)
        ref_hist_list.append(ref_hist)
        kl_divergence_sum = sum([stats.entropy(tar, ref) for tar, ref in zip(tar_hist, ref_hist)])
        tar_ref_score_list.append((idx, kl_divergence_sum))

    sorted_score_list = sorted(tar_ref_score_list, key=lambda x: x[1])
    # print(sorted_score_list)
    ref_idx_list = [ind for ind, val in sorted_score_list[:ref_cnt]]

    return ref_idx_list


def kernel_preprocessing(ker_path, scale, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')
    kernel = kernel_shift(kernel, scale)
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_preprocessing_without_pad(ker_path, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    # kernel = np.pad(kernel, 1, 'constant')
    kernel = kernel_shift(kernel, 1)
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_preprocessing_without_scale(ker_path, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')
    kernel = kernel_shift(kernel, 1)
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_preprocessing_without_shift(ker_path, scale, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')
    # kernel = kernel_shift(kernel, scale)
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_processing(ker_path, scale, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')
    kernel = kernel_shift(kernel, scale)
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_processing2(ker_path, scale, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')
    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def kernel_processing3(ker_path, scale, is_tensor=True):
    kernel = loadmat(ker_path)['Kernel']
    kernel = np.pad(kernel, 1, 'constant')

    if is_tensor:
        kernel = torch.FloatTensor(kernel).cuda()
        # kernel = torch.from_numpy(kernel).float().cuda()
    return kernel


def downscale_with_kernel(hr_img, kernel, stride=2):
    hr_img = make_1ch(hr_img)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    lr_img = F.conv2d(hr_img, kernel, stride=stride, padding=0)
    lr_img = make_3ch(lr_img)
    return lr_img


def kernel_shift(kernel, sf):
    current_center_of_mass = measurements.center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    shift_vec = wanted_center_of_mass - current_center_of_mass
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_ref_img(img_list, save_path):
    for i, img_path in enumerate(img_list):
        img = Image.open(img_path)
        # remove on 230810
        # w, h = img.size
        # img = img.resize((w // 4, h // 4))
        img.save(os.path.join(save_path, f"ref{i + 1}_" + os.path.basename(img_path)), optimize=True, quality=30)


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1 / sf), padding=padding)


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def calc_curr_k(dn_network_params):
    """given a generator network, the function calculates the kernel it is imitating"""
    curr_k = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(dn_network_params):
        curr_k = F.conv2d(curr_k, w, padding=w.shape[-1]-1)
    curr_k = curr_k.squeeze().flip([0, 1])
    return curr_k


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


# Source from DASR
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


class DegradationProcessing(object):
    def __init__(self,
                 conf,
                 kernel,
                 mode='bicubic'):

        self.scale = conf.scale
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

