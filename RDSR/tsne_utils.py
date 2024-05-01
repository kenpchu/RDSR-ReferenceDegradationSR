from glob import glob
from datetime import datetime

from options import Options, JsonOptions
# from train_options import Options

from networks.blindsr import make_model as make_sr_model
from utils.util import SRMDPreprocessing, BatchBlur
from matplotlib.legend_handler import HandlerTuple, HandlerPathCollection
from scipy.io import loadmat
from PIL import Image
from sklearn.manifold import TSNE

import os
import random
import imageio
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolor
import torch


class DegradeProcess(object):

    def __init__(self, config, lambda_1, lambda_2, theta):
        self.conf = config
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.degrade_fn = None
        self.init_degrade_fn()

    def init_degrade_fn(self):
        self.degrade_fn = SRMDPreprocessing(
            scale=self.conf.scale,
            kernel_size=self.conf.blur_kernel,
            blur_type=self.conf.blur_type,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            theta=self.theta,
            noise=self.conf.noise
        )

    def execute_degrade(self, hr_img):
        lr_img, b_kernels = self.degrade_fn(hr_img, random=False)

        return lr_img.cuda(), b_kernels


def visualization_tsne(embedded, save_path, tnse_name='tnse', labels=[]):
    # plt.figure(figsize=(5, 5))
    # base_color_list = ['b', 'g', 'k']
    base_color_list = list(mcolor.TABLEAU_COLORS.keys())

    color_sel_list = []
    if embedded.shape[0] > len(base_color_list):
        color_sel_list = cm.rainbow(np.linspace(0, 1, embedded.shape[0]-len(base_color_list)))

    if len(labels) == 0:
        labels = [str(i+1) for i in range(embedded.shape[0])]

    # colors list
    for i in range(embedded.shape[0]):
        if i < len(base_color_list):
            tmp_c = base_color_list[i]
        else:
            tmp_c = np.array(color_sel_list[i-len(base_color_list)]).reshape(1, -1)
        scatter = plt.scatter(embedded[i, 0, :, 0], embedded[i, 0, :, 1], c=tmp_c, label=labels[i])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    legend = plt.legend(fontsize=8, loc='lower right', ncol=1)

    # handles, _ = scatter.legend_elements()

    # # Set the alpha value for the legend handles
    # alpha = 0.4
    # for handle in handles:
    #     handle.set_alpha(alpha)

    # for text in legend.get_texts():
    #     text.set_alpha(alpha)
    print(os.path.join(save_path, f'tnse_{tnse_name}.png'))
    plt.savefig(os.path.join(save_path, f'tnse_{tnse_name}.png'))

    plt.close('all')
    pass


def evaluate_DR(config, encoder, output_path, tnse_name='tnse'):
    # lambda_1_list = [0.5, 4, 0.5, 3.7, 3.2, 1.5]
    # lambda_2_list = [0.5, 4, 3.0, 0.6, 2.4, 3.9]
    # theta_list = [0, 90, 70, 30, 135, 270]

    # lambda_1_list = [0.8, 2.0, 2.0, 3.2, 0.66]
    # lambda_2_list = [0.8, 3.0, 1.0, 1.5, 0.55]
    # theta_list =    [0  ,  70,  30, 135,  321]


    lambda_1_list = [0.8, 2.0, 2.0, 3.2]
    lambda_2_list = [0.8, 3.0, 1.0, 1.5]
    theta_list =    [0  ,  70,  30, 135]

    fea_list = []

    # TSNE testing image:
    hr_img_list = sorted(glob(os.path.join(config.tsne_img_path, "*.png")))
    assert len(hr_img_list) != 0

    sample_count = config.tsne_img_count
    if len(hr_img_list) < config.tsne_img_count:
        sample_count = len(hr_img_list)
    sample_indices_list = random.sample(range(len(hr_img_list)), sample_count)
    sample_indices_list = [i for i in range(config.tsne_img_count)]
    # sample_hr_img_list = [hr_img_list[i] for i in sample_indices_list]

    # TODO: add random lambda & theta to make sure TSNE result
    cus_kernel_cnt = 1
    for _ in range(cus_kernel_cnt):
        lambda_1 = np.random.rand() * (config.lambda_max - config.lambda_min) + config.lambda_min
        lambda_2 = np.random.rand() * (config.lambda_max - config.lambda_min) + config.lambda_min
        theta = np.random.rand() * 360
        print(f"lambda_1: {lambda_1}, lambda_2: {lambda_2}, theta: {theta}")
        lambda_1_list.append(lambda_1)
        lambda_2_list.append(lambda_2)
        theta_list.append(theta)

    assert len(lambda_1_list) == len(lambda_2_list)
    assert len(lambda_2_list) == len(theta_list)
    config.blur_kernel = 11

    encoder.eval()
    for lambda_1, lambda_2, theta in zip(lambda_1_list, lambda_2_list, theta_list):
        degrade_inst = DegradeProcess(config, lambda_1, lambda_2, theta)

        for sample_i in sample_indices_list:
            with torch.no_grad():
                # hr_img = imageio.imread(hr_img_list[sample_i])
                hr_img = Image.open(hr_img_list[sample_i]).convert('RGB')
                hr_img = np.array(hr_img, dtype=np.uint8)

                if np.ndim(hr_img) < 3:
                    hr_img = np.stack([hr_img, hr_img, hr_img], 2)
                hr_img = np.ascontiguousarray(hr_img.transpose((2, 0, 1)))
                hr_img = torch.from_numpy(hr_img).float().cuda().unsqueeze(0).unsqueeze(1)
                b, n, c, h, w = hr_img.size()
                hr_img = hr_img[:, :, :, :h // config.scale * config.scale, :w // config.scale * config.scale]

                lr_img, b_kernels = degrade_inst.execute_degrade(hr_img)
                b_kernels = b_kernels.cpu().numpy()

                _, fea = encoder.encoder_q(lr_img[:, 0, ...])
                # print(fea)
                # return
                fea_list.append(fea.data.cpu().numpy())
    encoder.train()

    f = np.concatenate(fea_list, 0)
    f_min = np.min(f, 0)
    f_max = np.max(f, 0)
    f_norm = (f - f_min) / (f_max - f_min)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedded = tsne.fit_transform(f)
    embedded = embedded.reshape(len(lambda_1_list), 1, len(sample_indices_list), -1)
    # print(embedded)
    visualization_tsne(embedded, output_path, tnse_name=tnse_name)


def evaluate_DR_test(config, encoder, output_path, tnse_name='tnse_test'):
    fea_list = []

    # TSNE testing image:
    hr_img_list = sorted(glob(os.path.join(config.tsne_img_path, "*.png")), key=lambda x: int(x.split('_')[1]))
    # print(hr_img_list)
    assert len(hr_img_list) != 0

    hr_img_list = hr_img_list[:config.tsne_img_count]
    print(len(hr_img_list))
    # ker_list = [os.path.join(config.kernel_gt_dir , "kernel_" + os.path.basename(hr_path).split('_')[1] + ".mat") for hr_path in hr_img_list]
    ker_list = sorted(glob(os.path.join(config.datasets_dir, config.kernel_gt_dir, "*.mat")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # print(config.kernel_gt_dir)
    # print(ker_list)
    ker_type_cnt = config.tsne_ker_count
    ker_list = ker_list[:ker_type_cnt]
    print(len(ker_list))
    # return
    encoder.encoder_q.eval()
    # for ker_i in range(ker_type_cnt):
    #     degrade_inst = DegradeProcess(config, lambda_1, lambda_2, theta)
    #     ker_idx = i % ker_type_cnt
    #     ker_path = ker_list[ker_i]
    for ker_i in range(len(ker_list)):

        ker_path = ker_list[ker_i]
        kernel = loadmat(ker_path)['Kernel']
        ker_t = torch.from_numpy(kernel).float().cuda()
        blur_fn = BatchBlur(kernel_size=kernel.shape[-1])

        for i in range(len(hr_img_list)):
            hr_img = Image.open(hr_img_list[i]).convert('RGB')
            hr_img = np.array(hr_img, dtype=np.uint8)
            hr_img = np.ascontiguousarray(hr_img.transpose((2, 0, 1)))
            hr_img = torch.from_numpy(hr_img).float().cuda().unsqueeze(0)
            # hr_img = torch.from_numpy(hr_img).float().cuda()

            B, C, H, W = hr_img.size()

            hr_blured = blur_fn(hr_img.view(B, -1, H, W), ker_t)
            lr_img = hr_blured.view(-1, C, H, W)  # BN, C, H, W
            with torch.no_grad():
                _, fea = encoder.encoder_q(lr_img)
            fea_list.append(fea.data.cpu().numpy())

    encoder.encoder_q.train()
    print(len(fea_list))

    f = np.concatenate(fea_list, 0)
    f_min = np.min(f, 0)
    f_max = np.max(f, 0)
    f_norm = (f - f_min) / (f_max - f_min)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedded = tsne.fit_transform(f)
    # embedded = embedded.reshape(ker_type_cnt, 1, len(hr_img_list) // ker_type_cnt, -1)
    embedded = embedded.reshape(ker_type_cnt, 1, len(hr_img_list), -1)

    visualization_tsne(embedded, output_path, tnse_name=tnse_name)


def evaluate_DR_test1(config, encoder, output_path, tnse_name=1, ker_st=0):
    fea_list = []

    # TSNE testing image:
    hr_img_list = sorted(glob(os.path.join(config.tsne_img_path, "*.png")), key=lambda x: int(x.split('_')[1]))

    assert len(hr_img_list) != 0

    hr_img_list = hr_img_list[:config.tsne_img_count]

    ker_list = [os.path.join(os.path.dirname(hr_path), "../gt_k_x2", "kernel_" + os.path.basename(hr_path).split('_')[1] + ".mat") for hr_path in hr_img_list]

    ker_type_cnt = conf.tsne_ker_count
    ker_sample_list = ker_list[ker_st:ker_st + ker_type_cnt]
    encoder.encoder_q.eval()

    for ker_i in range(ker_type_cnt):
        ker_path = ker_sample_list[ker_i]

        for i in range(len(hr_img_list)):
            hr_img = Image.open(hr_img_list[i]).convert('RGB')
            hr_img = np.array(hr_img, dtype=np.uint8)
            hr_img = np.ascontiguousarray(hr_img.transpose((2, 0, 1)))
            hr_img = torch.from_numpy(hr_img).float().cuda().unsqueeze(0)
            # hr_img = torch.from_numpy(hr_img).float().cuda()

            kernel = loadmat(ker_path)['Kernel']
            # ker_t = torch.from_numpy(kernel).float().cuda().unsqueeze(0)
            ker_t = torch.from_numpy(kernel).float().cuda()
            blur_fn = BatchBlur(kernel_size=kernel.shape[-1])

            B, C, H, W = hr_img.size()

            hr_blured = blur_fn(hr_img.view(B, -1, H, W), ker_t)
            lr_img = hr_blured.view(-1, C, H, W)  # BN, C, H, W
            with torch.no_grad():
                _, fea = encoder.encoder_q(lr_img)
            fea_list.append(fea.data.cpu().numpy())
    encoder.encoder_q.train()
    
    f = np.concatenate(fea_list, 0)
    f_min = np.min(f, 0)
    f_max = np.max(f, 0)
    f_norm = (f - f_min) / (f_max - f_min)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedded = tsne.fit_transform(f)
    embedded = embedded.reshape(ker_type_cnt, 1, len(hr_img_list), -1)

    visualization_tsne(embedded, output_path, tnse_name=tnse_name)


def evaluate_DR_test2(config, encoder, output_path, tnse_name=1, ker_st=0):
    fea_list = []

    # TSNE testing image:
    hr_img_list = sorted(glob(os.path.join(config.tsne_img_path, "*.png")), key=lambda x: int(x.split('_')[1]))

    assert len(hr_img_list) != 0

    hr_img_list = hr_img_list[:config.tsne_img_count]

    ker_list = [os.path.join(os.path.dirname(hr_path), "../cus_k_x2", "img_" + os.path.basename(hr_path).split('_')[1] + "_k_x2.mat") for hr_path in hr_img_list]

    ker_type_cnt = conf.tsne_ker_count
    ker_sample_list = ker_list[ker_st:ker_st + ker_type_cnt]
    encoder.encoder_q.eval()

    for ker_i in range(ker_type_cnt):
        ker_path = ker_sample_list[ker_i]

        for i in range(len(hr_img_list)):
            hr_img = Image.open(hr_img_list[i]).convert('RGB')
            hr_img = np.array(hr_img, dtype=np.uint8)
            hr_img = np.ascontiguousarray(hr_img.transpose((2, 0, 1)))
            hr_img = torch.from_numpy(hr_img).float().cuda().unsqueeze(0)
            # hr_img = torch.from_numpy(hr_img).float().cuda()

            kernel = loadmat(ker_path)['Kernel']
            # ker_t = torch.from_numpy(kernel).float().cuda().unsqueeze(0)
            ker_t = torch.from_numpy(kernel).float().cuda()
            blur_fn = BatchBlur(kernel_size=kernel.shape[-1])

            B, C, H, W = hr_img.size()

            hr_blured = blur_fn(hr_img.view(B, -1, H, W), ker_t)
            lr_img = hr_blured.view(-1, C, H, W)  # BN, C, H, W
            with torch.no_grad():
                _, fea = encoder.encoder_q(lr_img)
            fea_list.append(fea.data.cpu().numpy())
    encoder.encoder_q.train()

    f = np.concatenate(fea_list, 0)
    f_min = np.min(f, 0)
    f_max = np.max(f, 0)
    f_norm = (f - f_min) / (f_max - f_min)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedded = tsne.fit_transform(f)
    embedded = embedded.reshape(ker_type_cnt, 1, len(hr_img_list), -1)

    visualization_tsne(embedded, output_path, tnse_name=tnse_name)

    pass



if __name__ == '__main__':
    # opt = Options()
    opt = JsonOptions()
    conf = opt.get_config()
    tmp_scale = conf.scale
    conf.scale = [conf.scale]
    sr_model = make_sr_model(conf).cuda()
    conf.scale = tmp_scale
    if conf.pretrained_baseline_path:
        sr_model.load_state_dict(torch.load(conf.pretrained_baseline_path), strict=False)
    sr_model.eval()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(conf.output_dir, timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # evaluate_DR(conf, sr_model.E, save_path)
    # evaluate_DR_test(conf, sr_model.E, save_path)
    evaluate_DR_test1(conf, sr_model.E, save_path)

    # timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # save_path = os.path.join(conf.output_dir, timestamp)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # # random kernel
    # for i in range(conf.tsne_img_count // conf.tsne_ker_count):
    #     evaluate_DR_test1(conf, sr_model.E, save_path, tnse_name=i, ker_st=conf.tsne_ker_count * i)

    # timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # save_path = os.path.join(conf.output_dir, timestamp)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # # customerize kernel 
    # for i in range(conf.tsne_img_count // conf.tsne_ker_count):
    #     evaluate_DR_test2(conf, sr_model.E, save_path, tnse_name=i, ker_st=conf.tsne_ker_count * i)


    pass





