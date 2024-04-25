import argparse
import torch
import os
from util import create_train_logger2
from datetime import datetime
import json
from types import SimpleNamespace


class DoeOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--cfg_path', type=str, default='')
        self.parser.add_argument('--para_path', type=str, default='')

    def get_para_dict(self):
        option = self.parser.parse_args()
        with open(option.para_path) as json_fp:
            para_data_str = json_fp.read()
            para_data = json.loads(para_data_str)

        return para_data

    def load_config_list(self):
        option = self.parser.parse_args()
        conf_list = []
        if option.cfg_path and option.para_path:
            with open(option.cfg_path) as json_fp:
                cfg_data_str = json_fp.read()
                cfg_data = json.loads(cfg_data_str)
            with open(option.para_path) as json_fp:
                para_data_str = json_fp.read()
                para_data = json.loads(para_data_str)
            description_str = "description"
            description = para_data[description_str] if description_str in para_data.keys() else ""
            cfg_data[description_str] = description
            conf_list.append(cfg_data)
            for key in para_data.keys():
                if isinstance(para_data[key], list):
                    for val in para_data[key]:
                        cfg_update_data = cfg_data.copy()
                        cfg_update_data[key] = val
                        val_str = str(val).replace(".", "_")
                        cfg_update_data[description_str] = description + "_" + key + "_" + val_str
                        conf_list.append(cfg_update_data)

        return conf_list


class JsonOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--cfg_path', type=str, default='')

        # self.conf = self.parser.parse_args()
        self.conf = None

    def load_config(self):
        option = self.parser.parse_args()
        if option.cfg_path:
            with open(option.cfg_path) as json_fp:
                cfg_data = json_fp.read()
            cfg_obj = json.loads(cfg_data, object_hook=lambda d: SimpleNamespace(**d))
            self.conf = cfg_obj
        return self.set_base_env()

    def load_config_by_dict(self, cfg_dict):
        self.conf = SimpleNamespace(**cfg_dict)
        return self.set_base_env()

    def set_base_env(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.conf.timestamp = timestamp
        self.conf.output_dir_ori = os.path.join(self.conf.output_dir, timestamp)
        os.makedirs(self.conf.output_dir_ori)
        create_train_logger2(timestamp, self.conf.output_dir)

        if self.conf.colab:
            import colab

        return self.conf

    def get_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(os.path.basename(img_name))[0]
        str_list = os.path.splitext(img_name)[0].split('_')
        # kernel_name = 'kernel_' + '_'.join(str_list[1:])
        kernel_name = 'kernel_' + str_list[-1]
        # hr_name = 'img_' + '_'.join(str_list[1:])
        hr_name = 'img_' + str_list[-1]
        # print

        # print(hr_name)

        # self.conf.input_image_path = os.path.join(self.conf.input_dir, img_name)
        self.conf.input_image_path = img_name
        # self.conf.kernel_path = os.path.join(self.conf.kernel_dir, self.conf.abs_img_name + '.mat') if self.conf.kernel_dir != '' else None
        self.conf.kernel_path = os.path.join(self.conf.kernel_dir, kernel_name + '.mat') if self.conf.kernel_dir != '' else None
        # self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None
        self.conf.gt_path = os.path.join(self.conf.gt_dir, hr_name + '_gt.png') if self.conf.gt_dir != '' else None

        img_save_path = os.path.join(self.conf.output_dir_ori, self.conf.abs_img_name)
        os.makedirs(img_save_path)
        self.conf.output_dir = img_save_path

        if hasattr(self.conf, "up_pretrained_model") and self.conf.up_pretrained_model:
            self.conf.up_pre_model = os.path.join(self.conf.up_pretrained_model, f"{self.conf.abs_img_name}_sr_model.pt")
            self.conf.up_pre_model = os.path.join(self.conf.up_pretrained_model, f"model_sr_{self.conf.abs_img_name}_3000.pt")
            # self.conf.abs_img_name
            pass
        if hasattr(self.conf, "dn_pretrained_model") and self.conf.dn_pretrained_model:
            self.conf.dn_pre_model = os.path.join(self.conf.dn_pretrained_model, f"{self.conf.abs_img_name}_dn_model.pt")
            self.conf.dn_pre_model = os.path.join(self.conf.dn_pretrained_model, f"model_dn_{self.conf.abs_img_name}_3000.pt")
            pass

        print('*' * 60 + '\nRunning DualSR ...')
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        print('grand-truth kernel: \'%s\'' %self.conf.kernel_path)
        print('img_save_path: \'%s\'' % img_save_path)
        return self.conf

    def get_div2k_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(os.path.basename(img_name))[0]
        img_no = self.conf.abs_img_name[:4]
        kernel_name = 'kernel_' + img_no
        hr_name = img_no
        self.conf.input_image_path = img_name
        self.conf.kernel_path = os.path.join(self.conf.kernel_dir, kernel_name + '.mat') if self.conf.kernel_dir != '' else None
        self.conf.gt_path = os.path.join(self.conf.gt_dir, hr_name + '.png' ) if self.conf.gt_dir != '' else None

        img_save_path = os.path.join(self.conf.output_dir_ori, self.conf.abs_img_name)
        os.makedirs(img_save_path)
        self.conf.output_dir = img_save_path

        print('*' * 60 + '\nRunning DualSR ...')
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        print('grand-truth kernel: \'%s\'' %self.conf.kernel_path)
        print('img_save_path: \'%s\'' % img_save_path)
        return self.conf


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DualSR')

        # Paths
        self.parser.add_argument('--input_dir', '-i', type=str, default='test/LR', help='path to image input directory.')
        self.parser.add_argument('--output_dir', '-o', type=str, default='train_log', help='path to image output directory.')
        self.parser.add_argument('--kernel_dir', '-k', type=str, default='', help='path to grand-truth kernel directory.')
        self.parser.add_argument('--gt_dir', '-g', type=str, default='', help='path to grand-truth image.')
        self.parser.add_argument('--ref_dir', type=str, default='', help='path to grand-truth image.')
        self.parser.add_argument('--ref_select_path', type=str, default='')
        self.parser.add_argument('--target_select_path', type=str, default='')
        self.parser.add_argument('--ref_random', type=int, default=0)

        self.parser.add_argument('--input_image_path', default='', help='path to one specific image file')
        self.parser.add_argument('--kernel_path', default=None, help='path to one specific kernel file')
        self.parser.add_argument('--gt_path', default=None, help='path to one specific ground truth file')
        self.parser.add_argument('--ref_path_list', default=None, help='path to one specific ground truth file')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=128, help='crop size for HR patch')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
        self.parser.add_argument('--scale_factor', type=int, default=2, help='The upscaling scale factor')
        # self.parser.add_argument('--scale_factor_downsampler', type=float, default=0.5, help='scale factor for downsampler')
        # self.parser.add_argument('--scale_factor', type=int, default=4, help='The upscaling scale factor')
        self.parser.add_argument('--scale_factor_downsampler', type=float, default=0.5, help='scale factor for downsampler')

        #Lambda Parameters
        self.parser.add_argument('--lambda_cycle', type=int, default=5, help='lambda parameter for cycle consistency loss')
        self.parser.add_argument('--lambda_interp', type=int, default=2, help='lambda parameter for masked interpolation loss')
        self.parser.add_argument('--lambda_regularization', type=int, default=2, help='lambda parameter for downsampler regularization term')

        # Learning rates
        self.parser.add_argument('--lr_G_UP', type=float, default=0.001, help='initial learning rate for upsampler generator')
        self.parser.add_argument('--lr_G_DN', type=float, default=0.0002, help='initial learning rate for downsampler generator')
        self.parser.add_argument('--lr_D_DN', type=float, default=0.0002, help='initial learning rate for downsampler discriminator')
        self.parser.add_argument('--lr_E', type=float, default=0.0001, help='initial learning rate for downsampler discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')
        self.parser.add_argument('--step_size', type=int, default=750, help='Adam momentum')
        self.parser.add_argument('--gamma', type=float, default=0.5, help='Adam momentum')

        # Iterations
        # self.parser.add_argument('--num_iters', type=int, default=3000, help='number of training iterations')
        self.parser.add_argument('--num_iters', type=int, default=300, help='number of training iterations')
        # self.parser.add_argument('--eval_iters', type=int, default=200, help='for debug purpose')
        self.parser.add_argument('--eval_iters', type=int, default=100, help='for debug purpose')
        # self.parser.add_argument('--plot_iters', type=int, default=100, help='for debug purpose')
        self.parser.add_argument('--plot_iters', type=int, default=200, help='for debug purpose')
        self.parser.add_argument('--debug', action='store_true', help='plot intermediate results')
        self.parser.add_argument('--save_img', type=int, default=1)

        # ken add
        self.parser.add_argument('--lambda_target_ref_gan_ratio', type=float, default=0.8)
        self.parser.add_argument('--lambda_ref', type=float, default=1)
        self.parser.add_argument('--lambda_ref_cycle', type=float, default=1)
        self.parser.add_argument('--lambda_ref_vgg_loss', type=float, default=1)
        self.parser.add_argument('--lambda_target_vgg_loss', type=float, default=1)
        self.parser.add_argument('--lambda_hf_loss', type=float, default=1)
        self.parser.add_argument('--lambda_cx_loss', type=float, default=0)
        self.parser.add_argument('--lambda_dn_gan_loss', type=float, default=1)
        self.parser.add_argument('--lambda_ref_dn_gan_loss', type=float, default=1)
        self.parser.add_argument('--lambda_up_gan_loss', type=float, default=1)
        self.parser.add_argument('--lambda_w_regularization', type=int, default=1,
                                 help='lambda parameter for downsampler regularization term')
        self.parser.add_argument('--ref_baseline_ratio', type=float, default=0.8)
        self.parser.add_argument('--earlystop_iters', type=int, default=3000)
        self.parser.add_argument('--ref_stop_iters', type=int, default=2000)
        self.parser.add_argument('--ref_count', type=int, default=5)
        self.parser.add_argument('--target_ind', type=int, default=0)
        self.parser.add_argument('--target_count', type=int, default=2)
        self.parser.add_argument('--ref_continue_mode', type=int, default=0)
        self.parser.add_argument('--noise', type=float, default=0)
        self.parser.add_argument('--crop_ratio', type=float, default=0.5)
        self.parser.add_argument('--pretrained_encoder_path', type=str, default="../pre-trained_model/model_500_fine.pt")
        self.parser.add_argument('--with_dr', type=int, default=1)
        self.parser.add_argument('--start_dr_iter', type=int, default=1)
        self.parser.add_argument('--start_min_loss_iter', type=int, default=0)
        self.parser.add_argument('--UP_add_R', type=int, default=0)
        self.parser.add_argument('--DN_add_R', type=int, default=0)
        self.parser.add_argument('--colab', type=int, default=0)
        self.parser.add_argument('--force_bicubic_iters', type=int, default=500)

        self.conf = self.parser.parse_args()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.conf.timestamp = timestamp
        create_train_logger2(timestamp, self.conf.output_dir)
        self.conf.output_dir = os.path.join(self.conf.output_dir, f'{timestamp}')
        self.conf.output_dir_ori = self.conf.output_dir
        if not os.path.exists(self.conf.output_dir):
            # os.makedirs(self.conf.output_dir)
            # os.path.join("train_log", self.conf.output_dir + f'_{timestamp}')
            print(self.conf.output_dir)
            os.makedirs(self.conf.output_dir)

    def get_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(os.path.basename(img_name))[0]
        str_list = os.path.splitext(img_name)[0].split('_')
        # kernel_name = 'kernel_' + '_'.join(str_list[1:])
        kernel_name = 'kernel_' + str_list[-1]
        # hr_name = 'img_' + '_'.join(str_list[1:])
        hr_name = 'img_' + str_list[-1]
        # print

        # print(hr_name)

        self.conf.input_image_path = os.path.join(self.conf.input_dir, img_name)
        # self.conf.input_image_path = img_name

        # self.conf.kernel_path = os.path.join(self.conf.kernel_dir, self.conf.abs_img_name + '.mat') if self.conf.kernel_dir != '' else None
        self.conf.kernel_path = os.path.join(self.conf.kernel_dir, kernel_name + '.mat') if self.conf.kernel_dir != '' else None
        # self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None
        self.conf.gt_path = os.path.join(self.conf.gt_dir, hr_name + '_gt.png' ) if self.conf.gt_dir != '' else None

        # for different dataset
        # hr_name = self.conf.abs_img_name[:-2]
        # self.conf.gt_path = os.path.join(self.conf.gt_dir, hr_name + '.png') if self.conf.gt_dir != '' else None

        img_save_path = os.path.join(self.conf.output_dir_ori, os.path.basename(self.conf.abs_img_name))
        print(img_save_path)
        os.makedirs(img_save_path)
        self.conf.output_dir = img_save_path

        print('*' * 60 + '\nRunning DualSR ...')
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        print('grand-truth kernel: \'%s\'' %self.conf.kernel_path)
        print('img_save_path: \'%s\'' % img_save_path)
        return self.conf




