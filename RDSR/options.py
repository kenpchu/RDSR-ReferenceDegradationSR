import argparse
import os
from datetime import datetime

import json
from types import SimpleNamespace


class JsonOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--cfg_path', type=str, default='')

        # self.conf = self.parser.parse_args()
        self.config = None

    def get_config(self):
        option = self.parser.parse_args()
        if option.cfg_path:
            with open(option.cfg_path) as json_fp:
                cfg_data = json_fp.read()
            cfg_obj = json.loads(cfg_data, object_hook=lambda d: SimpleNamespace(**d))
            self.config = cfg_obj

        return self.config


class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # training settings
        self.parser.add_argument('--random_seed', type=int, default=0)
        self.parser.add_argument('--tb_name', type=str, default='TIME', help='Tensorboard logger directory')
        self.parser.add_argument('--exp_name', type=str, default='')
        # Size
        # self.parser.add_argument('--scale_f', type=int, default=2)
        self.parser.add_argument('--input_crop_size', type=int, default=128)
        self.parser.add_argument('--patch_size', type=int, default=48, help='output patch size')

        # Iterations
        self.parser.add_argument('--train_iters', type=int, default=3000)
        self.parser.add_argument('--train_encoder_iters', type=int, default=3000)
        self.parser.add_argument('--train_sr_iters', type=int, default=500)
        self.parser.add_argument('--train_epochs', type=int, default=300)

        # batch size
        self.parser.add_argument('--batch_size', type=int, default=1)

        # training datasets folder settings
        self.parser.add_argument('--datasets_dir', type=str, default='../datasets/RDSR_dataset', help='dataset workspace')
        self.parser.add_argument('--ref_dir', type=str, default='video_dataset_1/ref/hr', help='relative path on datasets dir')
        self.parser.add_argument('--ref_gt_dir', type=str, default='', help='relative path on datasets dir')
        self.parser.add_argument('--kernel_gt_dir', type=str, default='', help='relative path on datasets dir')

        self.parser.add_argument('--target_dir', type=str, default='DIV2KRK/lr_x2', help='relative path on datasets dir')
        self.parser.add_argument('--target_gt_dir', type=str, default='DIV2KRK/gt', help='relative path on datasets dir')

        self.parser.add_argument('--ref_count', type=int, default=2, help='max ref count')
        self.parser.add_argument('--ref_limit', type=int, default=60, help='max ref count')
        self.parser.add_argument('--ref_st_ind', type=int, default=100, help='max ref count')
        self.parser.add_argument('--dn_freeze', type=int, default=1, help='max ref count')

        # output folder
        self.parser.add_argument('--output_dir', '-o', type=str, default="")

        # GPU
        self.parser.add_argument('--cpu', type=int, default=0, help='use cpu only')

        # Optimizer
        self.parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'SGD'], help='Optimizer')

        # save model
        self.parser.add_argument('--save_model', type=int, default=1, help='save_model')
        self.parser.add_argument('--save_image', type=int, default=1, help='save images')
        self.parser.add_argument('--train_log', type=str, default='train_loggers', help='save images')

        # record training results, such as loss, learning rate
        self.parser.add_argument('--plot_iters', type=int, default=50, help='visualization for debug')
        # self.parser.add_argument('--plot_iters', type=int, default=20, help='visualization for debug')
        self.parser.add_argument('--scale_iters', type=int, default=10, help='record scale for analysis')
        self.parser.add_argument('--evaluate_iters', type=int, default=100, help='visualization for debug') # useless
        self.parser.add_argument('--best_thres', type=int, default=1500, help='visualization for debug') # useless
        self.parser.add_argument('--target_thres', type=int, default=1500, help='visualization for debug') # useless

        self.parser.add_argument('--target_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--target_lr_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--target_sr_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--ref_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--ref_lr_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--dr_lambda', type=float, default=10, help='visualization for debug') # useless
        self.parser.add_argument('--tv_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--interpo_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--hf_lambda', type=float, default=0, help='visualization for debug') # useless
        self.parser.add_argument('--color_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--vgg_tar_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--vgg_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--vgg_ref_lambda', type=float, default=0, help='visualization for debug')  # useless
        self.parser.add_argument('--gv_ref_lambda', type=float, default=0, help='visualization for debug')
        self.parser.add_argument('--dn_regular_lambda', type=float, default=0, help='visualization for debug')
        self.parser.add_argument('--gan_lambda', type=float, default=0, help='visualization for debug')
        self.parser.add_argument('--brisque_lambda', type=float, default=0, help='visualization for debug')
        self.parser.add_argument('--ref_dn_regular_lambda', type=float, default=0, help='visualization for debug')
        self.parser.add_argument('--total_target_lambda', type=float, default=1, help='visualization for debug') # useless
        self.parser.add_argument('--total_ref_lambda', type=float, default=1, help='visualization for debug') # useless

        # optimizer parameter
        # sr init lr is fine tuned.
        self.parser.add_argument('--lr_up', type=float, default=1e-3, help='initial learning rate for upsample')
        self.parser.add_argument('--lr_cor', type=float, default=1e-3, help='initial learning rate for upsample')
        self.parser.add_argument('--lr_up_disc', type=float, default=1e-5, help='initial learning rate for upsample discriminator')
        self.parser.add_argument('--lr_dn_disc', type=float, default=0.0002, help='initial learning rate for upsample discriminator')
        self.parser.add_argument('--lr_dn', type=float, default=1e-3, help='initial learning rate for downsample')
        # encoder init lr is fine tuned.
        self.parser.add_argument('--lr_en', type=float, default=2e-3, help='initial learning rate for encoder')
        self.parser.add_argument('--beta1', type=float, default=0.25, help='Adam momentum')

        # scheduler parameter
        self.parser.add_argument('--lrs_step_size', type=int, default=800, help='lrs step size')
        self.parser.add_argument('--lrs_gamma', type=float, default=0.5, help='lrs gamma')
        self.parser.add_argument('--lrs_milestone', type=str, default='100,300,500,900,1800', help='lrs gamma')

        # scheduler matrix parameter
        self.parser.add_argument('--lrs_matrix', type=int, default=0, help='patience of ReduceLROnPlateau')
        self.parser.add_argument('--lrs_patience', type=float, default=50, help='patience of ReduceLROnPlateau')
        self.parser.add_argument('--lrs_cooldown', type=float, default=100, help='cooldown of ReduceLROnPlateau')
        # factor refers to gamma
        self.parser.add_argument('--lrs_minlr', type=float, default=1e-6, help='mini lr of ReduceLROnPlateau')

        # DASR parameters
        # self.parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
        self.parser.add_argument('--scale', type=int, default=2, help='super resolution scale')
        self.parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
        # self.parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')

        self.parser.add_argument('--blur_kernel', type=int, default=21,
                            help='size of blur kernels')
        self.parser.add_argument('--blur_type', type=str, default='aniso_gaussian',
                            help='blur types (iso_gaussian | aniso_gaussian)')
        self.parser.add_argument('--mode', type=str, default='bicubic',
                            help='downsampler (bicubic | s-fold)')
        self.parser.add_argument('--noise', type=float, default=0.0,
                            help='noise level')
        self.parser.add_argument('--sig_min', type=float, default=0.2,
                            help='minimum sigma of isotropic Gaussian blurs')
        self.parser.add_argument('--sig_max', type=float, default=4.0,
                            help='maximum sigma of isotropic Gaussian blurs')
        self.parser.add_argument('--lambda_min', type=float, default=0.2,
                            help='minimum value for the eigenvalue of anisotropic Gaussian blurs')
        self.parser.add_argument('--lambda_max', type=float, default=4.0,
                            help='maximum value for the eigenvalue of anisotropic Gaussian blurs')
        self.parser.add_argument('--random_kernel', type=int, default=1, help='is server')

        self.parser.add_argument('--server', type=int, default=0, help='is server')
        self.parser.add_argument('--ref_pool', type=int, default=0, help='is server')
        self.parser.add_argument('--pretrained_lr_path', type=str, default='', help='is server')
        self.parser.add_argument('--pretrained_baseline_path', type=str, default='../pre-trained_model/model_500_fine.pt', help='is server')
        self.parser.add_argument('--pretrained_encoder_path', type=str, default='../pre-trained_model/model_500_fine.pt', help='is server')

        self.parser.add_argument('--tsne_img_path', type=str, default='../datasets/DIV2K_train_HR')
        self.parser.add_argument('--tsne_img_count', type=int, default=100)
        self.parser.add_argument('--tsne_ker_count', type=int, default=10)

        self.parser.add_argument('--target_ind', type=int, default=0)
        self.parser.add_argument('--target_count', type=int, default=0)
        self.parser.add_argument('--ref_selection', type=int, default=0)
        self.parser.add_argument('--rgb_range', type=float, default=1.0)
        self.parser.add_argument('--is_normalize', type=int, default=1)

        self.parser.add_argument('--ref_continue_mode', type=int, default=0)

        self.parser.add_argument('--ref_select_path', type=str, default='')
        self.parser.add_argument('--target_select_path', type=str, default='')
        self.parser.add_argument('--crop_ratio', type=float, default=1.0)

        self.conf = self.parser.parse_args()
        if self.conf.lrs_milestone:
            milestone_list = self.conf.lrs_milestone.split(',')
            self.conf.lrs_milestone = [int(mile) for mile in milestone_list]

    def get_config(self):
        return self.conf
