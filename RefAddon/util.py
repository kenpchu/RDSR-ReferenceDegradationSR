import os
import sys
import torch
import logging
import numpy as np
from PIL import Image
import scipy.io as sio
import json
import random
import math
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t, normalize_en = False):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.transpose(move2cpu(im_t[0]), (1, 2, 0))  
    if normalize_en:
        im_np = (im_np + 1.0) / 2.0
    im_np = np.clip(np.round(im_np * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def im2tensor(im_np, normalize_en = False):
    """Copy the image to the gpu & converts to range [-1,1]"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    if normalize_en:
        im_np = im_np * 2.0 - 1.0
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).unsqueeze(0).cuda()


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1 / sf), padding=padding)
    
  
def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def make_1ch(im):
    s = im.shape
    assert s[1] == 3
    return im.reshape(s[0] * 3, 1, s[2], s[3])


def make_3ch(im):
    s = im.shape
    assert s[1] == 1
    return im.reshape(s[0] // 3, 3, s[2], s[3])


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    
    assert (a.shape[r] >= b.shape[r]) and (a.shape[c] >= b.shape[c])
    assert ((a.shape[r] - b.shape[r]) % 2 == 0) and ((a.shape[c] - b.shape[c]) % 2 == 0)
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min
    

def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    #kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_final_kernel(k_2, conf):
    """saves the final kernel the results folder"""
    sio.savemat(os.path.join(conf.output_dir, '%s_kernel_x2.mat' % conf.abs_img_name), {'Kernel': k_2})


def calc_curr_k(G_DW_params):
    """given a generator network, the function calculates the kernel it is imitating"""
    curr_k = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(G_DW_params):
        curr_k = F.conv2d(curr_k, w, padding=w.shape[-1]-1) #if ind == 0 else F.conv2d(curr_k, w)
    curr_k = curr_k.squeeze().flip([0, 1])
    return curr_k

      
def downscale_with_kernel(hr_img, kernel, stride=2):
    hr_img = make_1ch(hr_img)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    #padding = (kernel.shape[-1] - 1) // 2
    lr_img = F.conv2d(hr_img, kernel, stride=stride, padding=0)
    lr_img = make_3ch(lr_img)
    return lr_img


def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


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
    
    Y_A = Y_A[border:-border,border:-border]
    Y_B = Y_B[border:-border,border:-border]
    
    e=Y_A-Y_B;
    mse=np.mean(e**2);
    psnr_cur=10*np.log10(255*255/mse);
    
    return psnr_cur


def calculate_psnr(img1_np, img2_np):
    '''
    This calculation is from IKENet
    '''
    # img1 and img2 have range [0, 255]
    img1 = img1_np.astype(np.float64)
    img2 = img2_np.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def np2tensor(im_np, normalize_en=False, rgb_range=255.0):
    # Copy the image to the gpu & converts to range [-1,1]
    np_transpose = np.ascontiguousarray(im_np.transpose((2, 0, 1)))
    img_np = np_transpose * (rgb_range / 255.0)
    if normalize_en:
        img_np = img_np * 2.0 - 1.0
    im_t = torch.from_numpy(img_np).float().cuda()

    return im_t


def create_train_logger2(timestamp, train_log_name):
    if not os.path.exists(f'{train_log_name}/{timestamp}'):
        os.makedirs(f'{train_log_name}/{timestamp}')
    main_logger = logging.getLogger(f"main")
    f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/main_{timestamp}.log")
    f_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    main_logger.setLevel(logging.DEBUG)
    main_logger.addHandler(f_handler)
    main_logger.addHandler(stream_handler)

    eval_logger = logging.getLogger(f"eval")
    eval_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/eval_{timestamp}.csv")
    eval_f_handler.setLevel(logging.DEBUG)
    eval_logger.setLevel(logging.DEBUG)
    eval_logger.addHandler(eval_f_handler)

    loss_w_logger = logging.getLogger(f"loss_w")
    loss_w_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/loss_w_{timestamp}.csv")
    loss_w_f_handler.setLevel(logging.DEBUG)
    loss_w_logger.setLevel(logging.DEBUG)
    loss_w_logger.addHandler(loss_w_f_handler)

    loss_logger = logging.getLogger(f"loss")
    loss_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/loss_{timestamp}.csv")
    loss_f_handler.setLevel(logging.DEBUG)
    loss_logger.setLevel(logging.DEBUG)
    loss_logger.addHandler(loss_f_handler)

    lr_logger = logging.getLogger(f"lr")
    lr_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/lr_{timestamp}.csv")
    lr_f_handler.setLevel(logging.DEBUG)
    lr_logger.setLevel(logging.DEBUG)
    lr_logger.addHandler(lr_f_handler)

    lr_logger.info(f'iteration, sr_lr, dn_lr, d_dn_lr, {timestamp}')
    loss_logger.info(f'iteration, loss_cycle_forward, loss_cycle_backward, loss_GAN, loss_regularization, total_loss, {timestamp}')
    loss_w_logger.info(f'iteration, loss_tar_lr, loss_tar_sr_gt, loss_tar_lr_gt, {timestamp}')
    eval_logger.info(f'iteration, tar_hr_psnr, tar_lr_psnr, tar_lr_gt_psnr, {timestamp}')


def close_train_logger():
    logger_name_list = ["main", "eval", "loss_w", "loss", "lr"]
    for name in logger_name_list:
        main_logger = logging.getLogger(name)
        for handler in main_logger.handlers[:]:
            main_logger.removeHandler(handler)
            handler.close()


def dump_training_settings(conf):
    # dump config to dictionary
    conf_dict = dict()
    for arg in vars(conf):
        conf_dict[arg] = getattr(conf, arg)

    with open(os.path.join(conf.output_dir_ori, 'config.json'), 'w') as conf_fp:
        json.dump(conf_dict, conf_fp, indent=4)


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

def calculate_rgb_mean(img_path):
    im = read_image(img_path)
    r_mean = np.mean(im[0])
    g_mean = np.mean(im[1])
    b_mean = np.mean(im[2])
    return r_mean, g_mean, b_mean

def ref_preprocessing(conf, ref_path_list):
    random.seed(conf.ref_random)

    ref_sample_path_list = ref_path_list
    # TODO: add function to select reference sample images
    # if conf.div2k:
    #     ref_idx_list = sample_ref_by_color_space(conf.input_image_path, ref_sample_path_list, ref_cnt=conf.ref_count)
    if conf.ref_random:
        ref_idx_list = random.sample(range(len(ref_sample_path_list)), conf.ref_count)
    else:
        ref_idx_list = sample_ref_by_color_space(conf.input_image_path, ref_sample_path_list, ref_cnt=conf.ref_count)
    ref_list = [ref_sample_path_list[idx] for idx in ref_idx_list]

    print(ref_list)

    return ref_list


def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = move2cpu(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def analytic_kernel_w(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = torch.zeros((3 * k_size - 2, 3 * k_size - 2)).cuda()
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()
