import torch
import loss
import networks
import util
import os
import math
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from PIL import Image
from torch_sobel import Sobel
from datetime import datetime
from util import shave_a2b, tensor2im
from brisque import BRISQUE
from networks1 import Generator_UP_DRCA, Generator_DN_DRCA


class DualSR:
    lambda_update_freq = 200
    bic_loss_to_start_change = 0.4
    lambda_bicubic_decay_rate = 100.
    update_l_rate_freq = 750
    update_l_rate_rate = 4.
    lambda_sparse_end = 5
    lambda_centralized_end = 1
    lambda_bicubic_min = 5e-6

    def __init__(self, conf, test_loader=None, encoder_network=None):
        # Fix random seed
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
    
        # Acquire configuration
        self.conf = conf
        self.filename = os.path.basename(conf.input_image_path).split('.')[0]
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.test_loader = test_loader
        
        # Define the networks

        self.G_DN = networks.Generator_DN().cuda()
        self.D_DN = networks.Discriminator_DN().cuda()
        self.G_UP = networks.Generator_UP().cuda()
        if self.conf.scale_factor == 4:
            # 25.55654 vs 25.49742 crop:128 -> 25.64558
            self.G_DN = networks.Generator_DN_x4().cuda()
            # self.G_DN = networks.Generator_DN_x4_Kernel(self.G_DN).cuda()
            self.G_UP = networks.Generator_UP_x4().cuda()

        self.E = None
        self.G_R_DN = None
        self.G_R_UP = None
        if self.conf.with_dr:
            self.E = encoder_network.cuda()
            self.G_R_DN = Generator_DN_DRCA(self.G_DN).cuda()
            self.G_R_UP = Generator_UP_DRCA(self.G_UP, scale_factor=self.conf.scale_factor).cuda()
            # self.G_R_DN = Generator_R_DN().cuda()
            # self.G_R_UP = networks.Generator_R_DN().cuda()

        # Losses
        self.criterion_gan = loss.GANLoss().cuda()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_interp = torch.nn.L1Loss()
        self.l1_loss = loss.CharbonnierLoss().cuda()
        self.vgg_loss = loss.VGG('22', rgb_range=1).cuda()
        self.regularization = loss.DownsamplerRegularization(conf.scale_factor_downsampler, self.G_DN.G_kernel_size)

        # Initialize networks weights
        self.G_DN.apply(networks.weights_init_G_DN)
        self.D_DN.apply(networks.weights_init_D_DN)
        self.G_UP.apply(networks.weights_init_G_UP)
        if self.conf.with_dr:
            self.G_R_DN.apply(networks.weights_init_G_DN)
            self.G_R_UP.apply(networks.weights_init_G_UP)

        # Optimizers
        self.optimizer_G_DN = torch.optim.Adam(self.G_DN.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))
        self.optimizer_D_DN = torch.optim.Adam(self.D_DN.parameters(), lr=conf.lr_D_DN, betas=(conf.beta1, 0.999))
        self.optimizer_G_UP = torch.optim.Adam(self.G_UP.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
        self.optimizer_G_R_DN = None
        self.optimizer_G_R_UP = None
        if self.conf.with_dr:
            self.optimizer_E = torch.optim.Adam(
                self.E.parameters(), lr=conf.lr_E, betas=(conf.beta1, 0.999))
            self.optimizer_G_R_DN = torch.optim.Adam(
                self.G_R_DN.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))
            self.optimizer_G_R_UP = torch.optim.Adam(
                self.G_R_UP.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
            self.scheduler_E = torch.optim.lr_scheduler.StepLR(
                self.optimizer_E, step_size=self.conf.step_size, gamma=self.conf.gamma)
            self.scheduler_G_R_DN = torch.optim.lr_scheduler.StepLR(
                self.optimizer_G_R_DN, step_size=self.conf.step_size, gamma=1/self.update_l_rate_rate)
            self.scheduler_G_R_UP = torch.optim.lr_scheduler.StepLR(
                self.optimizer_G_R_UP, step_size=self.conf.step_size, gamma=self.conf.gamma)

        # Read input image

        self.in_img = util.read_image(conf.input_image_path)
        self.in_img_t = util.im2tensor(self.in_img)
        b_x = self.in_img_t.shape[2] % conf.scale_factor
        b_y = self.in_img_t.shape[3] % conf.scale_factor
        self.in_img_cropped_t = self.in_img_t[..., b_x:, b_y:]
        
        self.gt_img = util.read_image(conf.gt_path) if conf.gt_path is not None else None
        self.gt_kernel = loadmat(conf.kernel_path)['Kernel'] if conf.kernel_path is not None else None

        self.gt_kernel_t = None
        if self.gt_kernel is not None:
            self.gt_kernel = np.pad(self.gt_kernel, 1, 'constant')
            self.gt_kernel = util.kernel_shift(self.gt_kernel, sf=conf.scale_factor)
            self.gt_kernel_t = torch.FloatTensor(self.gt_kernel).cuda()
        
            self.gt_downsampled_img_t = util.downscale_with_kernel(self.in_img_cropped_t, self.gt_kernel_t)
            self.gt_downsampled_img = util.tensor2im(self.gt_downsampled_img_t)

            # Define other networks
            self.DN_with_gt = networks.DownScale(self.gt_kernel_t)
            self.degradation_with_gt = networks.DegradationProcessing(self.conf, self.gt_kernel_t)

        # Debug variables
        self.debug_steps = []
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None

        self.UP_psnrs = []
        self.DN_psnrs = []
        self.DN_gt_psnrs = []
        self.target_cycle_losses = []
        self.min_loss_psnr = 0
        self.min_loss_iter = -1

        if self.conf.debug:
            self.loss_GANs = []
            self.loss_cycle_forwards = []
            self.loss_cycle_backwards = []
            self.loss_interps = []
            self.loss_Discriminators = []

        self.iter = 0

        self.max_psnr = 0
        self.max_psnr_iter = 0
        self.record_str = ''
        self.earlystop_iters = conf.earlystop_iters

        self.logger = None
        self.eval_logger = None
        self.loss_w_logger = None
        self.loss_logger = None
        self.lr_logger = None
        self.init_loggers()

        self.set_test_input()

        # learner parameters
        self.bic_loss_counter = 0
        self.loss_regularization_w = 0
        self.loss_interp_w = 0
        self.similar_to_bicubic = False  # Flag indicating when the bicubic similarity is achieved
        self.insert_constraints = True  # Flag is switched to false once constraints are added to the loss

        self.G_UP_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G_UP, step_size=self.conf.step_size, gamma=self.conf.gamma)

        self.ref_baseline_losses = []
        self.logger.info(f"{self.conf.output_dir}")

        self.brisque_metric = BRISQUE()

    def init_loggers(self):
        self.logger = logging.getLogger('main')
        self.eval_logger = logging.getLogger('eval')
        self.loss_w_logger = logging.getLogger('loss_w')
        self.loss_logger = logging.getLogger('loss')
        self.lr_logger = logging.getLogger('lr')
        self.logger.info(f'{self.filename},')
        self.eval_logger.info(f'{self.filename},')
        self.loss_logger.info(f'{self.filename},')
        self.loss_w_logger.info(f'{self.filename},')
        self.lr_logger.info(f'{self.filename},')

    def set_train_mode(self):
        self.G_DN.train()
        self.G_UP.train()
        if self.conf.with_dr:
            self.G_R_DN.train()
            self.G_R_UP.train()
            self.E.train()
            pass

    def set_eval_mode(self):
        self.G_DN.eval()
        self.G_UP.eval()
        if self.conf.with_dr:
            self.G_R_DN.eval()
            self.G_R_UP.eval()
            self.E.eval()
        pass

    @staticmethod
    def clean_cuda_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def set_input(self, data):
        self.real_HR = data['HR']
        self.real_LR = data['LR']
        self.real_LR_bicubic = data['LR_bicubic']

        self.data_dict = data
        # remove
        # self.tar_lr_g_b = data['tar_big_patch']
        # self.tar_lr_d_s = data['tar_small_patch']
        # self.tar_lr_s_bq = data['tar_small_bq_up']

        self.ref_hr_g_b = data['ref_big_patch']
        self.ref_hr_bq_dn_up = data['ref_big_bq_dn_up']
        self.ref_hr_d_s = data['ref_small_patch']
        self.ref_hr_s_bq = data['ref_small_bq_up']

    def set_test_input(self, is_last=0):
        test_data = next(iter(self.test_loader))

        self.tar_lr_w = test_data['Target_Img']
        self.tar_hr_gt_w = test_data['Target_Gt']
        self.tar_lr_bqup_w = test_data['Target_Img_Bq']
        self.ref_imgs_w = test_data['Ref_Imgs']
        self.ref_bq_imgs_w = test_data['Ref_Bq_Imgs']

        if self.conf.crop_ratio > 0:
            self.tar_lr_w = self.crop_img_t(self.tar_lr_w, self.conf.crop_ratio)
            self.tar_hr_gt_w = self.crop_img_t(self.tar_hr_gt_w, self.conf.crop_ratio)
            self.tar_lr_bqup_w = self.crop_img_t(self.tar_lr_bqup_w, self.conf.crop_ratio)

            for ind, ref_img in enumerate(self.ref_imgs_w):
                self.ref_imgs_w[ind] = self.crop_img_t(self.ref_imgs_w[ind], self.conf.crop_ratio)
                self.ref_bq_imgs_w[ind] = self.crop_img_t(self.ref_bq_imgs_w[ind], self.conf.crop_ratio)

        if is_last:
            self.tar_lr_w = test_data['Target_Img']
            self.tar_hr_gt_w = test_data['Target_Gt']
            self.tar_lr_bqup_w = test_data['Target_Img_Bq']


    @staticmethod
    def crop_img_t(img, ratio):
        n, c, h, w = img.shape
        h_ind = math.floor((1 - ratio) / 2.0 * h)
        h_length = math.floor(h * ratio)
        w_ind = math.floor((1 - ratio) / 2.0 * w)
        w_length = math.floor(w * ratio)
        return img[:, :, h_ind:h_ind + h_length, w_ind:w_ind + w_length]

    def train_v2(self, data):
        self.set_train_mode()
        self.set_input(data)

        # Forward with Degradation Representation
        if self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            # self.train_dr_G()
            self.train_G_dr()
            self.train_D_ref()
        else:
            self.train_G_0()
            self.train_D_0()

        if self.conf.with_dr:
            if self.iter > self.conf.start_dr_iter:
                self.update_learner_dr_v2()
            else:
                self.update_learner_v2()
        else:
            self.update_learner()

        self.iter = self.iter + 1
        if self.iter % self.conf.eval_iters == 0:
            self.eval_whole_img()
            self.log_learning_rate()
            self.logger.info(f'iteration: {self.iter}')

        self.update_lambda()

    def train(self, data):
        self.set_train_mode()
        self.set_input(data)

        # Forward with Degradation Representation
        if self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            # self.train_dr_G()
            self.train_G_dr()
            self.train_D()
        else:
            self.train_G()
            self.train_D()

        if self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            self.update_learner_dr()
        else:
            self.update_learner()

        self.iter = self.iter + 1
        if self.iter % self.conf.eval_iters == 0:
            self.eval_whole_img()
            self.log_learning_rate()
            self.logger.info(f'iteration: {self.iter}')

        self.update_lambda()

    def train_G_dr(self):
        # Turn off gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], False)

        # Rese gradient valus
        self.optimizer_G_R_DN.zero_grad()
        self.optimizer_G_R_UP.zero_grad()
        self.optimizer_E.zero_grad()

        # Forward path
        tar_d_dr, _, _ = self.E(self.real_LR, self.real_LR)
        self.fake_HR = self.G_R_UP(self.real_LR, tar_d_dr)
        self.rec_LR = self.G_R_DN(self.fake_HR, tar_d_dr)
        # Backward path
        tar_g_dr, _, _ = self.E(self.real_HR, self.real_HR)
        self.fake_LR = self.G_R_DN(self.real_HR, tar_g_dr)
        self.rec_HR = self.G_R_UP(self.fake_LR, tar_g_dr)

        self.total_loss = 0

        # Losses
        self.loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
        self.loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR,
                                                                                   self.rec_LR)) * self.conf.lambda_cycle
        self.loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR,
                                                                                    self.rec_HR)) * self.conf.lambda_cycle

        sobel_A = Sobel()(self.real_LR_bicubic.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)

        self.loss_interp = self.criterion_interp(self.fake_HR * loss_map_A,
                                                 self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp

        self.curr_k = util.calc_curr_k(self.G_DN.parameters())
        if self.conf.scale_factor == 4:
            self.curr_k = self.G_DN.calc_ker()
        self.loss_regularization = self.regularization(self.curr_k, self.real_HR,
                                                       self.fake_LR) * self.conf.lambda_regularization

        self.total_loss += self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization

        self.loss_ref_GAN = 0
        self.loss_ref_cycle = 0
        self.loss_ref_baseline = 0

        if self.conf.lambda_ref:
            # ref fake LR for discriminator & cal GAN loss
            self.ref_fake_LR = self.G_DN(self.ref_hr_g_b)
            self.loss_ref_GAN = self.criterion_gan(self.D_DN(self.ref_fake_LR), True)
            self.total_loss += self.loss_ref_GAN

            if self.conf.lambda_ref_cycle and self.iter < self.conf.ref_stop_iters:
                # ref cycle loss
                ref_dr, _, _ = self.E(self.ref_fake_LR, self.ref_fake_LR)
                self.ref_rec_hr = self.G_R_UP(self.ref_fake_LR, ref_dr)
                self.loss_ref_cycle = self.l1_loss(self.ref_rec_hr, shave_a2b(self.ref_hr_g_b, self.ref_rec_hr))
                # ref cycle baseline
                self.loss_ref_baseline = self.l1_loss(self.ref_hr_bq_dn_up, self.ref_hr_g_b)
                loss_ref_baseline = self.loss_ref_baseline * self.conf.ref_baseline_ratio
                # self.ref_rec_hr = self.G_UP(self.ref_fake_LR)
                if self.loss_ref_cycle > loss_ref_baseline:
                    self.total_loss += self.loss_ref_cycle * self.conf.lambda_ref_cycle
                pass

            # self.ref_rec_hr = self.G_UP(self.ref_fake_LR)
            # self.loss_ref_cycle = self.l1_loss(self.ref_rec_hr, shave_a2b(self.ref_hr_g_b, self.ref_rec_hr))
            # self.total_loss += self.loss_ref_cycle * self.conf.lambda_ref

        self.total_loss.backward()

        self.optimizer_G_R_UP.step()
        self.optimizer_G_R_DN.step()
        self.optimizer_E.step()

        if self.iter % self.conf.eval_iters == 0:
            self.loss_logger.info(f'{self.iter}, {format(self.loss_cycle_forward, ".5f")}, '
                                  f'{format(self.loss_cycle_backward, ".5f")}, {format(self.loss_GAN, ".5f")}, '
                                  f'{format(self.loss_regularization, ".5f")}, {format(self.loss_ref_GAN, ".5f")}, '
                                  f'{format(self.loss_ref_cycle, ".5f")}, {format(self.loss_ref_baseline, ".5f")}, '
                                  f'{format(self.regularization.loss_bicubic, ".5f")}, {format(self.total_loss, ".5f")}')

    def train_dr_G_0(self):
        util.set_requires_grad([self.D_DN], False)
        self.optimizer_G_R_DN.zero_grad()
        self.optimizer_G_R_UP.zero_grad()
        self.tar_lr = self.real_HR
        # TODO: encoder input with rgb_range 255?
        dr = self.E(self.tar_lr, self.tar_lr)
        tar_hr_rec = self.G_R_UP(self.tar_lr, dr)
        tar_lr_rec = self.G_R_DN(tar_hr_rec, dr)

        self.total_loss = 0
        # target loss
        loss_tar = self.l1_loss(tar_lr_rec, shave_a2b(self.tar_lr, tar_lr_rec))
        loss_tar_vgg = self.vgg_loss(tar_lr_rec, shave_a2b(self.tar_lr, tar_lr_rec))
        total_tar_loss = loss_tar + loss_tar_vgg
        self.total_loss += total_tar_loss

        # TODO: how to get ref dr? from ref_hr? from ref DN ?
        ref_dn_rec = self.G_DN(self.ref_hr_g_b)
        ref_dr = self.E(ref_dn_rec, ref_dn_rec)
        ref_lr_rec = self.G_R_DN(self.ref_hr_g_b, ref_dr)
        ref_hr_rec = self.G_R_UP(ref_lr_rec, ref_dr)

        # ref loss
        ref_base_loss = self.l1_loss(self.ref_hr_g_b, self.ref_hr_bq_dn_up)
        ref_base_loss += self.vgg_loss(self.ref_hr_g_b, self.ref_hr_bq_dn_up)

        loss_ref = self.l1_loss(ref_hr_rec, shave_a2b(self.ref_hr_g_b, ref_hr_rec))
        loss_ref += self.vgg_loss(ref_hr_rec, shave_a2b(self.ref_hr_g_b, ref_hr_rec))
        if loss_ref > ref_base_loss * self.conf.ref_baseline_ratio:
            self.total_loss += loss_ref

        self.total_loss.backward()

        self.optimizer_G_R_DN.step()
        self.optimizer_G_R_UP.step()

        if self.iter % self.conf.eval_iters == 0:
            self.loss_logger.info(f'{self.iter}, {format(loss_tar, ".5f")}, {format(loss_tar_vgg, ".5f")}, '
                                  f'{format(loss_ref, ".5f")}, {format(ref_base_loss, ".5f")}')

    def train_G_0(self):
        # Turn off gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], False)

        # Rese gradient valus
        self.optimizer_G_UP.zero_grad()
        self.optimizer_G_DN.zero_grad()

        # Forward path
        self.fake_HR = self.G_UP(self.real_LR)
        self.rec_LR = self.G_DN(self.fake_HR)
        # Backward path
        self.fake_LR = self.G_DN(self.real_HR)
        self.rec_HR = self.G_UP(self.fake_LR)

        self.total_loss = 0

        # Losses
        self.loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
        self.loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR,
                                                                                   self.rec_LR)) * self.conf.lambda_cycle
        self.loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR,
                                                                                    self.rec_HR)) * self.conf.lambda_cycle

        sobel_A = Sobel()(self.real_LR_bicubic.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        self.loss_interp = self.criterion_interp(self.fake_HR * loss_map_A,
                                                 self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp

        self.curr_k = util.calc_curr_k(self.G_DN.parameters())
        if self.conf.scale_factor == 4:
            self.curr_k = self.G_DN.calc_ker()
        self.loss_regularization = self.regularization(self.curr_k, self.real_HR,
                                                       self.fake_LR) * self.conf.lambda_regularization

        self.total_loss += self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization

        self.loss_ref_GAN = 0
        self.loss_ref_cycle = 0
        self.loss_ref_baseline = 0

        self.total_loss.backward()

        self.optimizer_G_UP.step()
        self.optimizer_G_DN.step()

        if self.iter % self.conf.eval_iters == 0:
            self.loss_logger.info(f'{self.iter}, {format(self.loss_cycle_forward, ".5f")}, '
                                  f'{format(self.loss_cycle_backward, ".5f")}, {format(self.loss_GAN, ".5f")}, '
                                  f'{format(self.loss_regularization, ".5f")}, {format(self.loss_ref_GAN, ".5f")}, '
                                  f'{format(self.loss_ref_cycle, ".5f")}, {format(self.loss_ref_baseline, ".5f")}, '
                                  f'{format(self.regularization.loss_bicubic, ".5f")}, {format(self.total_loss, ".5f")}')

    def train_G(self):
        # Turn off gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], False)

        # Rese gradient valus
        self.optimizer_G_UP.zero_grad()
        self.optimizer_G_DN.zero_grad()

        # Forward path
        self.fake_HR = self.G_UP(self.real_LR)
        self.rec_LR = self.G_DN(self.fake_HR)
        # Backward path
        self.fake_LR = self.G_DN(self.real_HR)
        self.rec_HR = self.G_UP(self.fake_LR)

        self.total_loss = 0

        # Losses
        self.loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
        self.loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) * self.conf.lambda_cycle
        self.loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) * self.conf.lambda_cycle
        
        sobel_A = Sobel()(self.real_LR_bicubic.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        self.loss_interp = self.criterion_interp(self.fake_HR * loss_map_A, self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp
        
        self.curr_k = util.calc_curr_k(self.G_DN.parameters())
        if self.conf.scale_factor == 4:
            self.curr_k = self.G_DN.calc_ker()
        self.loss_regularization = self.regularization(self.curr_k, self.real_HR, self.fake_LR) * self.conf.lambda_regularization

        self.total_loss += self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization

        self.loss_ref_GAN = 0
        self.loss_ref_cycle = 0
        self.loss_ref_baseline = 0

        if self.conf.lambda_ref:
            # ref fake LR for discriminator & cal GAN loss
            self.ref_fake_LR = self.G_DN(self.ref_hr_g_b)
            self.loss_ref_GAN = self.criterion_gan(self.D_DN(self.ref_fake_LR), True)
            self.total_loss += self.loss_ref_GAN

            if self.conf.lambda_ref_cycle and self.iter < self.conf.ref_stop_iters:
                # ref cycle loss
                self.ref_rec_hr = self.G_UP(self.ref_fake_LR)
                self.loss_ref_cycle = self.l1_loss(self.ref_rec_hr, shave_a2b(self.ref_hr_g_b, self.ref_rec_hr))
                # ref cycle baseline
                self.loss_ref_baseline = self.l1_loss(self.ref_hr_bq_dn_up, self.ref_hr_g_b)
                loss_ref_baseline = self.loss_ref_baseline * self.conf.ref_baseline_ratio
                # self.ref_rec_hr = self.G_UP(self.ref_fake_LR)

                self.total_loss += self.loss_ref_cycle * self.conf.lambda_ref_cycle
                pass

            # self.ref_rec_hr = self.G_UP(self.ref_fake_LR)
            # self.loss_ref_cycle = self.l1_loss(self.ref_rec_hr, shave_a2b(self.ref_hr_g_b, self.ref_rec_hr))
            # self.total_loss += self.loss_ref_cycle * self.conf.lambda_ref

        self.total_loss.backward()
        
        self.optimizer_G_UP.step()
        self.optimizer_G_DN.step()

        if self.iter % self.conf.eval_iters == 0:
            self.loss_logger.info(f'{self.iter}, {format(self.loss_cycle_forward, ".5f")}, '
                                  f'{format(self.loss_cycle_backward, ".5f")}, {format(self.loss_GAN, ".5f")}, '
                                  f'{format(self.loss_regularization, ".5f")}, {format(self.loss_ref_GAN, ".5f")}, '
                                  f'{format(self.loss_ref_cycle, ".5f")}, {format(self.loss_ref_baseline, ".5f")}, '
                                  f'{format(self.regularization.loss_bicubic, ".5f")}, {format(self.total_loss, ".5f")}')

    def train_D(self):
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)
        
        # Rese gradient valus
        self.optimizer_D_DN.zero_grad()

        # Fake
        pred_fake = self.D_DN(self.fake_LR.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Real
        pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
        loss_D_real = self.criterion_gan(pred_real, True)
        self.fake_LR.detach()
        # Combined loss and calculate gradients

        # ref discriminator
        # ref fake
        if self.conf.lambda_ref:
            pred_ref_fake = self.D_DN(self.ref_fake_LR.detach())
            loss_ref_D_fake = self.criterion_gan(pred_ref_fake, False)
            # ref real
            pred_ref_real = self.D_DN(util.shave_a2b(self.ref_hr_d_s, self.ref_fake_LR))
            loss_ref_D_real = self.criterion_gan(pred_ref_real, True)

        if self.conf.lambda_ref:
            self.loss_Discriminator = ((loss_D_real + loss_D_fake) * 0.5 * self.conf.lambda_target_ref_gan_ratio +
                                       (loss_ref_D_real + loss_ref_D_fake) * 0.5 * (1 - self.conf.lambda_target_ref_gan_ratio))
        else:
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
        self.loss_Discriminator.backward()

        # Update weights
        self.optimizer_D_DN.step()

    def train_D_0(self):
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)

        # Rese gradient valus
        self.optimizer_D_DN.zero_grad()

        # Fake
        pred_fake = self.D_DN(self.fake_LR.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Real
        pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
        loss_D_real = self.criterion_gan(pred_real, True)
        self.fake_LR.detach()
        # Combined loss and calculate gradients

        self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
        self.loss_Discriminator.backward()

        # Update weights
        self.optimizer_D_DN.step()

    def train_D_ref(self):
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)

        # Rese gradient valus
        self.optimizer_D_DN.zero_grad()

        # Fake
        pred_fake = self.D_DN(self.fake_LR.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Real
        pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
        loss_D_real = self.criterion_gan(pred_real, True)
        self.fake_LR.detach()
        # Combined loss and calculate gradients

        # ref discriminator
        # ref fake
        if self.conf.lambda_ref and self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            pred_ref_fake = self.D_DN(self.ref_fake_LR.detach())
            loss_ref_D_fake = self.criterion_gan(pred_ref_fake, False)
            # ref real
            pred_ref_real = self.D_DN(util.shave_a2b(self.ref_hr_d_s, self.ref_fake_LR))
            loss_ref_D_real = self.criterion_gan(pred_ref_real, True)

        if self.conf.lambda_ref and self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            self.loss_Discriminator = ((loss_D_real + loss_D_fake) * 0.5 * self.conf.lambda_target_ref_gan_ratio +
                                       (loss_ref_D_real + loss_ref_D_fake) * 0.5 * (
                                                   1 - self.conf.lambda_target_ref_gan_ratio))
        else:
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
        self.loss_Discriminator.backward()

        # Update weights
        self.optimizer_D_DN.step()

    def save_model(self, image_info_str, is_best=False, save_model=False, iter_cnt=0):
        if not is_best:
            date_time = datetime.now()
            str_date_time = date_time.strftime("%Y%d%m-%H%M%S")
            image_info_str = f"{image_info_str}_{str_date_time}"

        iter_att = 'best' if is_best else 'last'
        if iter_cnt > 0:
            iter_att = str(iter_cnt)

        rec_img_name = f"{image_info_str}_{iter_att}_rec.png"
        plt.imsave(os.path.join(self.conf.output_dir, rec_img_name), self.upsampled_img)

        up_model_name = f"{image_info_str}_UP_{iter_att}.pt"
        down_model_name = f"{image_info_str}_DOWN_{iter_att}.pt"

        if is_best:
            up_model_name = f"{image_info_str}_{self.max_psnr_iter}_UP_{iter_att}.pt"
            down_model_name = f"{image_info_str}_{self.max_psnr_iter}_DOWN_{iter_att}.pt"
            if os.path.exists(os.path.join(self.conf.output_dir, up_model_name)):
                os.remove(os.path.join(self.conf.output_dir, up_model_name))
            if os.path.exists(os.path.join(self.conf.output_dir, down_model_name)):
                os.remove(os.path.join(self.conf.output_dir, down_model_name))

            up_model_name = f"{image_info_str}_{self.iter}_UP_{iter_att}.pt"
            down_model_name = f"{image_info_str}_{self.iter}_DOWN_{iter_att}.pt"

            self.max_psnr_iter = self.iter

        if save_model:
            torch.save(self.G_UP.state_dict(), os.path.join(self.conf.output_dir, up_model_name))
            torch.save(self.G_DN.state_dict(), os.path.join(self.conf.output_dir, down_model_name))
            pass
        # record all max psnr image
        rec_img_name = f"{image_info_str}_{self.iter}_rec.png"
        plt.imsave(os.path.join(self.conf.output_dir, rec_img_name), self.upsampled_img)

        self.record_str += f"{image_info_str}, {self.iter}, {self.UP_psnrs[-1]}\n"
        if not is_best and iter_cnt == 0:
            self.record_str += f"Best PSNR iter: {self.max_psnr_iter}, best psnr: {self.max_psnr}\n"
            csv_path = os.path.join(self.csv_path, f'{image_info_str}.csv')
            with open(csv_path, 'w') as fp:
                fp.write(self.record_str)

    def save_test_loader_image(self):
        for test_data in self.test_loader:
            for key in test_data.keys():
                if isinstance(test_data[key], list):
                    for i, ref in enumerate(test_data[key]):
                        self.save_img(ref, f"{key.lower()}_{i+1}")
                else:
                    self.save_img(test_data[key], f'{key.lower()}')

    def save_input_img(self, data_dict):
        for data_name in data_dict.keys():
            self.save_img(data_dict[data_name], f'{data_name.lower()}')

    def save_img(self, img, img_name):
        save_path = os.path.join(self.conf.output_dir, img_name + '.png')
        save_patch_img = Image.fromarray(util.tensor2im(img))
        save_patch_img.save(save_path)

    def log_learning_rate(self):
        cur_e_lr = 0

        if self.conf.with_dr and self.iter > self.conf.start_dr_iter:
            cur_dn_lr = self.optimizer_G_R_DN.param_groups[0]['lr']
            cur_up_lr = self.optimizer_G_R_UP.param_groups[0]['lr']
            cur_e_lr = self.optimizer_E.param_groups[0]['lr']
        else:
            cur_dn_lr = self.optimizer_G_DN.param_groups[0]['lr']
            cur_up_lr = self.optimizer_G_UP.param_groups[0]['lr']
        cur_dn_d_lr = self.optimizer_D_DN.param_groups[0]['lr']

        if self.iter % self.conf.eval_iters == 0:
            self.lr_logger.info(f'{self.iter}, {cur_up_lr}, {cur_dn_lr}, {cur_dn_d_lr}, {cur_e_lr}')
            self.lr_logger.info(f'{self.iter}, {self.regularization.lambda_bicubic}, {self.regularization.lambda_centralized}, {self.regularization.lambda_sparse}')

    def eval_ref_w_baseline(self):
        self.set_eval_mode()
        self.ref_baseline_losses = []

        with torch.no_grad():
            for ind, ref_img_w in enumerate(self.ref_imgs_w):
                ref_bq_w = self.ref_bq_imgs_w[ind]
                ref_baseline_loss = self.l1_loss(ref_bq_w, ref_img_w)
                self.ref_baseline_losses.append(ref_baseline_loss)
        ref_baseline_losses = [format(ref_loss, ".5f")  for ref_loss in self.ref_baseline_losses]
        ref_baseline_str = ', '.join(ref_baseline_losses)

        self.loss_w_logger.info(f'{self.iter}-w, {ref_baseline_str}')

    def eval_whole_img(self):
        self.set_eval_mode()
        self.clean_cuda_memory()

        self.ref_rec_lr_list = []
        self.ref_rec_hr_list = []
        self.ref_rec_losses = []
        ref_rec_losses = []

        loss_tar_lr = 0
        loss_tar_sr_gt = 0
        loss_tar_lr_gt = 0
        tar_hr_psnr = 0
        tar_lr_psnr = 0
        tar_lr_gt_psnr = 0

        if self.iter == self.conf.num_iters:
            self.set_test_input(is_last=1)

        with torch.no_grad():
            if self.conf.with_dr and self.iter > self.conf.start_dr_iter:
                tar_dr = self.E(self.tar_lr_w, self.tar_lr_w)
                tar_rec_hr_w = self.G_R_UP(self.tar_lr_w, tar_dr)
                tar_lr_dn_w = self.G_R_DN(self.tar_lr_w, tar_dr)
            else:
                tar_rec_hr_w = self.G_UP(self.tar_lr_w)
                tar_lr_dn_w = self.G_DN(self.tar_lr_w)
            self.tar_rec_hr_w = tar_rec_hr_w

            sobel_A = Sobel()(self.tar_lr_bqup_w.detach())
            loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
            # self.loss_interp_w = self.criterion_interp(tar_rec_hr_w * loss_map_A, self.tar_lr_bqup_w * loss_map_A)
            self.loss_interp_w = self.criterion_interp(tar_rec_hr_w * shave_a2b(loss_map_A, tar_rec_hr_w),
                                                       shave_a2b(self.tar_lr_bqup_w, tar_rec_hr_w) * shave_a2b(
                                                           loss_map_A, tar_rec_hr_w))
            self.curr_k = util.calc_curr_k(self.G_DN.parameters())

            loss_bicubic_w = 0
            # if self.conf.crop_ratio == 1:
            if self.conf.crop_ratio == 1 and self.conf.scale_factor != 4:
                self.loss_regularization_w = self.regularization(self.curr_k, self.tar_lr_w, tar_lr_dn_w)
                loss_bicubic_w = self.regularization.loss_bicubic

        if self.conf.crop_ratio < 1 and self.iter == self.conf.num_iters:
            tar_rec_hr_w_np = tensor2im(tar_rec_hr_w)
            tar_hr_gt_w_np = tensor2im(self.tar_hr_gt_w)
            tar_hr_psnr = util.cal_y_psnr(tar_rec_hr_w_np, tar_hr_gt_w_np, border=self.conf.scale_factor)
            self.UP_psnrs.append(tar_hr_psnr)
            self.eval_logger.info(f'{self.iter}, {format(tar_hr_psnr, ".5f")}')
            return

        # extra code
        tar_rec_hr_w_np = tensor2im(tar_rec_hr_w)
        brisque_score = self.brisque_metric.score(tar_rec_hr_w_np)
        tar_hr_gt_w_np = tensor2im(self.tar_hr_gt_w)
        tar_hr_psnr = util.cal_y_psnr(tar_rec_hr_w_np, shave_a2b(tar_hr_gt_w_np, tar_rec_hr_w_np), border=self.conf.scale_factor)
        self.UP_psnrs.append(tar_hr_psnr)

        # calculate reference whole loss
        ref_rec_avg_loss = 0
        # if self.iter != self.conf.num_iters:

        with torch.no_grad():
            self.clean_cuda_memory()
            tar_rec_lr_w = self.G_DN(tar_rec_hr_w)
            self.tar_rec_lr_w = tar_rec_lr_w
            # add GT for observation
            tar_gt_rec_lr_w = self.G_DN(self.tar_hr_gt_w)
            self.clean_cuda_memory()

            ref_rec_avg_loss = 0
            if not self.conf.ref_stop_iters:
                for ind, ref_img_w in enumerate(self.ref_imgs_w):
                    # TODO:start from here 1125
                    # ref_bq_w = self.ref_bq_imgs_w[ind]
                    # ref_baseline_loss = self.l1_loss(ref_bq_w, ref_img_w)
                    ref_rec_lr_w = self.G_DN(ref_img_w)
                    ref_rec_hr_w = self.G_UP(ref_rec_lr_w)

                    ref_bq_loss = self.regularization.criterion_bicubic(ref_img_w, ref_rec_lr_w)

                    self.ref_rec_lr_list.append(ref_rec_lr_w)
                    self.ref_rec_hr_list.append(ref_rec_hr_w)

                    # cal ref loss
                    ref_rec_loss = self.l1_loss(ref_rec_hr_w, shave_a2b(ref_img_w, ref_rec_hr_w))
                    ref_rec_losses.append(ref_rec_loss)
                    ref_rec_losses.append(ref_bq_loss)

                    # TODO: start from here
                    # add GT for observation
                    if self.gt_kernel is not None:
                        ref_lr_dn_gtk_w = self.DN_with_gt(ref_img_w)
                        ref_lr_de_gtk_w = self.degradation_with_gt(ref_img_w)
                        # cal ref gt loss
                        loss_ref_lr_gt_dn = self.l1_loss(shave_a2b(ref_lr_dn_gtk_w, ref_rec_lr_w), ref_rec_lr_w)
                        loss_ref_lr_gt_de = self.l1_loss(shave_a2b(ref_lr_de_gtk_w, ref_rec_lr_w), ref_rec_lr_w)

                ref_rec_avg_loss = sum(ref_rec_losses) / len(self.ref_imgs_w)

        loss_tar_lr = self.criterion_cycle(tar_rec_lr_w, shave_a2b(self.tar_lr_w, tar_rec_lr_w))

        # add GT for observation
        loss_tar_lr_gt = self.criterion_cycle(tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, tar_gt_rec_lr_w))
        loss_tar_sr_gt = self.criterion_cycle(tar_rec_hr_w, shave_a2b(self.tar_hr_gt_w, tar_rec_hr_w))

        # cal_psnr
        tar_rec_lr_w_np = tensor2im(tar_rec_lr_w)
        tar_lr_w_np = tensor2im(shave_a2b(self.tar_lr_w, tar_rec_lr_w))
        # tar_rec_hr_w_np = tensor2im(tar_rec_hr_w)
        # tar_hr_gt_w_np = tensor2im(self.tar_hr_gt_w)
        tar_gt_rec_lr_w_np = tensor2im(tar_gt_rec_lr_w)

        tar_lr_psnr = util.cal_y_psnr(tar_rec_lr_w_np, tar_lr_w_np, border=self.conf.scale_factor)
        # tar_hr_psnr = util.cal_y_psnr(tar_rec_hr_w_np, tar_hr_gt_w_np, border=self.conf.scale_factor)
        tar_lr_gt_psnr = util.cal_y_psnr(tar_gt_rec_lr_w_np, tar_lr_w_np, border=self.conf.scale_factor)

        self.DN_psnrs.append(tar_lr_psnr)
        self.DN_gt_psnrs.append(tar_lr_gt_psnr)
        if self.iter > self.conf.start_min_loss_iter:
            self.target_cycle_losses.append(loss_tar_lr)

        if max(self.UP_psnrs) == tar_hr_psnr:
            self.max_psnr = tar_hr_psnr
            self.max_psnr_iter = self.iter
            if self.conf.save_img:
                self.save_img(tar_rec_hr_w, 'tar_rec_hr_max_psnr_w')
            # self.save_all_model(prefix='max_sr_psnr')

        if len(self.target_cycle_losses) > 0 and min(self.target_cycle_losses) == loss_tar_lr:
            self.min_loss_psnr = tar_hr_psnr
            self.min_loss_iter = self.iter
            self.save_img(tar_rec_hr_w, 'tar_rec_hr_min_loss_w')

        # if self.iter > self.conf.ref_stop_iters and self.iter % (self.conf.eval_iters * 5) == 0:
        if self.iter > self.conf.start_min_loss_iter and self.iter % (self.conf.eval_iters * 5) == 0:
            if self.conf.save_img:
                self.save_img(tar_rec_hr_w, f'tar_rec_hr_{self.iter}_w')

        self.loss_w_logger.info(f'{self.iter}-w, {format(loss_tar_lr, ".5f")}, '
                                f'{format(loss_tar_sr_gt, ".5f")}, {format(loss_tar_lr_gt, ".5f")}, '
                                f'{format(self.loss_interp_w, ".5f")}, {format(self.loss_regularization_w, ".5f")}, '
                                f'{format(loss_bicubic_w, ".5f")}, '
                                f'{format(ref_rec_avg_loss, ".5f")}, {format(brisque_score, ".5f")}')

        self.eval_logger.info(f'{self.iter}, {format(tar_hr_psnr, ".5f")}, {format(tar_lr_psnr, ".5f")}, '
                              f'{format(tar_lr_gt_psnr, ".5f")}')

    def finish(self):
        util.save_final_kernel(util.move2cpu(self.curr_k), self.conf)
        self.save_img(self.tar_rec_hr_w, f'tar_rec_hr_{self.iter}_w')
        # self.save_img(self.tar_rec_lr_w, f'tar_rec_lr_{self.iter}_w')

        # save model
        # self.save_all_model()
        self.eval_logger.info(f'min_loss_psnr, max_sr_psnr, last_psnr, min_loss_iter, max_sr_iter')
        max_sr_psnr = format(self.max_psnr, ".5f")
        min_loss_psnr = format(self.min_loss_psnr, ".5f")
        last_sr_psnr = format(self.UP_psnrs[-1], ".5f")
        self.eval_logger.info(f'{min_loss_psnr}, {max_sr_psnr}, {last_sr_psnr}, {self.min_loss_iter}, {self.max_psnr_iter}')

        return self.min_loss_psnr

    def save_all_model(self, prefix=''):
        if prefix:
            dn_model_path = os.path.join(self.conf.output_dir, f'model_dn_{self.filename}_{prefix}.pt')
            sr_model_path = os.path.join(self.conf.output_dir, f'model_sr_{self.filename}_{prefix}.pt')
        else:
            dn_model_path = os.path.join(self.conf.output_dir, f'model_dn_{self.filename}_{self.iter}.pt')
            sr_model_path = os.path.join(self.conf.output_dir, f'model_sr_{self.filename}_{self.iter}.pt')

        torch.save(self.G_DN.state_dict(), dn_model_path)
        torch.save(self.G_UP.state_dict(), sr_model_path)

    def eval(self, is_final=False):
        self.quick_eval()
        if self.conf.debug:
            self.plot()

        util.save_final_kernel(util.move2cpu(self.curr_k), self.conf)
        plt.imsave(os.path.join(self.conf.output_dir, '%s.png' % self.conf.abs_img_name), self.upsampled_img)

        # if is_final:
        #     self.save_model(self.conf.abs_img_name)

        if self.gt_img is not None:
            print('Upsampler PSNR = ', self.UP_psnrs[-1])
            self.logger.info(f'Upsampler PSNR = {self.UP_psnrs[-1]}')
        if self.gt_kernel is not None:
            print("Downsampler PSNR = ", self.DN_psnrs[-1])
        print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.output_dir)
        plt.close('all')

    def quick_eval(self):
        # Evaluate trained upsampler and downsampler on input data
        with torch.no_grad():
            downsampled_img_t = self.G_DN(self.in_img_cropped_t)
            upsampled_img_t = self.G_UP(self.in_img_t)

        self.downsampled_img = util.tensor2im(downsampled_img_t)
        self.upsampled_img = util.tensor2im(upsampled_img_t)

        if self.gt_kernel is not None:
            self.DN_psnrs += [
                util.cal_y_psnr(self.downsampled_img, self.gt_downsampled_img, border=self.conf.scale_factor)]
        if self.gt_img is not None:
            self.UP_psnrs += [util.cal_y_psnr(self.upsampled_img, self.gt_img, border=self.conf.scale_factor)]
        self.debug_steps += [self.iter]

        if self.conf.debug:
            # Save loss values for visualization
            self.loss_GANs += [util.move2cpu(self.loss_GAN)]
            self.loss_cycle_forwards += [util.move2cpu(self.loss_cycle_forward)]
            self.loss_cycle_backwards += [util.move2cpu(self.loss_cycle_backward)]
            self.loss_interps += [util.move2cpu(self.loss_interp)]
            self.loss_Discriminators += [util.move2cpu(self.loss_Discriminator)]

        if self.earlystop_iters > 0 and self.iter > self.earlystop_iters:
            if self.max_psnr < self.UP_psnrs[-1]:
                self.save_model(self.conf.abs_img_name, is_best=True)
            if self.iter % (self.conf.num_iters // 3) == 0:
                self.save_model(self.conf.abs_img_name, is_best=False, iter_cnt=self.iter)
            self.max_psnr = max(self.UP_psnrs)

    def update_learner_dr(self):
        tmp_iter = self.iter
        if tmp_iter == 0:
            return

        # self.G_UP_lr_scheduler.step()
        self.scheduler_E.step()
        self.scheduler_G_R_DN.step()
        self.scheduler_G_R_UP.step()

        # Update learning rate every update_l_rate freq
        if tmp_iter % self.update_l_rate_freq == 0:
            # for params in self.optimizer_G_DN.param_groups:
            #     params['lr'] /= self.update_l_rate_rate
            for params in self.optimizer_D_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            # TODO: modify to evaluate whole image
            if self.iter % self.conf.eval_iters == 0:
                if self.regularization.loss_bicubic < self.bic_loss_to_start_change:
                    if self.bic_loss_counter >= 2:
                        self.similar_to_bicubic = True
                        self.logger.info(f'similar_to_bicubic, iter={tmp_iter}')
                    else:
                        self.bic_loss_counter += 1
                else:
                    self.bic_loss_counter = 0

            '''
            if self.regularization.loss_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                    self.logger.info(f'similar_to_bicubic, iter={tmp_iter}')
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
            '''
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif tmp_iter % self.lambda_update_freq == 0 and self.regularization.lambda_bicubic > self.lambda_bicubic_min:
            self.regularization.lambda_bicubic = max(self.regularization.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and self.regularization.lambda_bicubic < 5e-3:
                self.regularization.lambda_centralized = self.lambda_centralized_end
                self.regularization.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False

    def update_learner_dr_v2(self):
        tmp_iter = self.iter
        if tmp_iter == 0:
            return

        # self.G_UP_lr_scheduler.step()
        self.scheduler_E.step()
        self.scheduler_G_R_DN.step()
        self.scheduler_G_R_UP.step()

        # Update learning rate every update_l_rate freq
        if tmp_iter % self.update_l_rate_freq == 0:
            # for params in self.optimizer_G_DN.param_groups:
            #     params['lr'] /= self.update_l_rate_rate
            for params in self.optimizer_D_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if self.iter > self.conf.force_bicubic_iters:
                self.similar_to_bicubic = True
                self.logger.info(f'over {self.conf.force_bicubic_iters} iter, set similar_to_bicubic')
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif tmp_iter % self.lambda_update_freq == 0 and self.regularization.lambda_bicubic > self.lambda_bicubic_min:
            self.regularization.lambda_bicubic = max(self.regularization.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and self.regularization.lambda_bicubic < 5e-3:
                self.regularization.lambda_centralized = self.lambda_centralized_end
                self.regularization.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False

    def update_learner_v2(self):
        # self.scheduler_G_R_DN.step()
        # self.scheduler_G_R_UP.step()

        self.G_UP_lr_scheduler.step()
        tmp_iter = self.iter
        if tmp_iter == 0:
            return
        # Update learning rate every update_l_rate freq
        if tmp_iter % self.update_l_rate_freq == 0:
            for params in self.optimizer_G_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate
            for params in self.optimizer_D_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if self.iter > self.conf.force_bicubic_iters:
                self.similar_to_bicubic = True
                self.logger.info(f'over {self.conf.force_bicubic_iters} iter, set similar_to_bicubic')
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif tmp_iter % self.lambda_update_freq == 0 and self.regularization.lambda_bicubic > self.lambda_bicubic_min:
            self.regularization.lambda_bicubic = max(self.regularization.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and self.regularization.lambda_bicubic < 5e-3:
                self.regularization.lambda_centralized = self.lambda_centralized_end
                self.regularization.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False

    def update_learner(self):
        # self.scheduler_G_R_DN.step()
        # self.scheduler_G_R_UP.step()

        self.G_UP_lr_scheduler.step()
        tmp_iter = self.iter
        if tmp_iter == 0:
            return
        # Update learning rate every update_l_rate freq
        if tmp_iter % self.update_l_rate_freq == 0:
            for params in self.optimizer_G_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate
            for params in self.optimizer_D_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic and self.iter > self.conf.force_bicubic_iters:
            self.similar_to_bicubic = True
            self.logger.info(f'over {self.conf.force_bicubic_iters} iter, set similar_to_bicubic')

        if not self.similar_to_bicubic:
            if self.regularization.loss_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                    self.logger.info(f'similar_to_bicubic, iter={tmp_iter}')
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif tmp_iter % self.lambda_update_freq == 0 and self.regularization.lambda_bicubic > self.lambda_bicubic_min:
            self.regularization.lambda_bicubic = max(self.regularization.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and self.regularization.lambda_bicubic < 5e-3:
                self.regularization.lambda_centralized = self.lambda_centralized_end
                self.regularization.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False

    def update_lambda(self):
        pass
        # if self.iter <= self.ref_stop_iters:
        #     self.tmp_lambda_ref_cycle = self.conf.lambda_ref_cycle
        # elif self.ref_stop_iters < self.iter < self.conf.num_iters:
        #     self.conf.lambda_ref_cycle = 0
        # elif self.iter == self.conf.num_iters:
        #     self.conf.lambda_ref_cycle = self.tmp_lambda_ref_cycle

    def plot(self):
        loss_names = ['loss_GANs', 'loss_cycle_forwards', 'loss_cycle_backwards', 'loss_interps', 'loss_Discriminators']
        
        if self.gt_img is not None:
            plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                               in zip([self.UP_psnrs, self.DN_psnrs],
                                      ['Upsampler PSNR', 'Downsampler PSNR']) if x is not None])
        else:
            plots_data, labels = [0.0], 'None'
        
        plots_data2, labels2 = zip(*[(np.array(x), l) for (x, l)
                                   in zip([getattr(self, name) for name in loss_names],
                                          loss_names) if x is not None])
        # For the first iteration create the figure
        if not self.iter:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9, 8))
            #self.fig = plt.figure()
            grid = GridSpec(4, 4)
            self.psnr_plot_space = plt.subplot(grid[0:2, 0:2])
            self.loss_plot_space = plt.subplot(grid[0:2, 2:4])
            
            self.real_LR_space = plt.subplot(grid[2, 0])
            self.fake_HR_space = plt.subplot(grid[2, 1])
            self.rec_LR_space = plt.subplot(grid[2, 2])
            self.real_HR_space = plt.subplot(grid[3, 0])
            self.fake_LR_space = plt.subplot(grid[3, 1])
            self.rec_HR_space = plt.subplot(grid[3, 2])
            self.curr_ker_space = plt.subplot(grid[2, 3])
            self.ideal_ker_space = plt.subplot(grid[3, 3])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.psnr_plot_space.set_ylabel('db')
            self.psnr_plot_space.grid(True)
            self.psnr_plot_space.legend(labels)
            
            self.loss_plot_space.grid(True)
            self.loss_plot_space.legend(labels2)
            
            self.curr_ker_space.title.set_text('estimated kernel')
            self.ideal_ker_space.title.set_text('gt kernel')
            self.real_LR_space.title.set_text('$x$')
            self.real_HR_space.title.set_text('$y$')
            self.fake_HR_space.title.set_text('$G_{UP}(x)$')
            self.fake_LR_space.title.set_text('$G_{DN}(y)$')
            self.rec_LR_space.title.set_text('$G_{DN}(G_{UP}(x))$')
            self.rec_HR_space.title.set_text('$G_{UP}(G_{DN}(y))$')
            
            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.psnr_plot_space.plot(*[[0]] * 2 * len(plots_data))
            self.plots2 = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data2))
            
            # These line are needed in order to see the graphics at real time
            self.fig.tight_layout()
            self.fig.canvas.draw()
            # plt.pause(0.01)
            return

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.debug_steps, plot_data)
            
        for plot, plot_data in zip(self.plots2, plots_data2):
            plot.set_data(self.debug_steps, plot_data)
        
        self.psnr_plot_space.set_xlim([0, self.iter + 1])
        all_losses = np.array(plots_data)
        self.psnr_plot_space.set_ylim([np.min(all_losses)*0.9, np.max(all_losses)*1.1])
        
        self.loss_plot_space.set_xlim([0, self.iter + 1])
        all_losses2 = np.array(plots_data2)
        self.loss_plot_space.set_ylim([np.min(all_losses2)*0.9, np.max(all_losses2)*1.1])

        self.psnr_plot_space.legend(labels)
        self.loss_plot_space.legend(labels2)

        # Show current images
        self.curr_ker_space.imshow(util.move2cpu(self.curr_k))
        if self.gt_kernel is not None:
            self.ideal_ker_space.imshow(self.gt_kernel)
        self.real_LR_space.imshow(util.tensor2im(self.real_LR))
        self.real_HR_space.imshow(util.tensor2im(self.real_HR))
        self.fake_HR_space.imshow(util.tensor2im(self.fake_HR))
        self.fake_LR_space.imshow(util.tensor2im(self.fake_LR))
        self.rec_LR_space.imshow(util.tensor2im(self.rec_LR))
        self.rec_HR_space.imshow(util.tensor2im(self.rec_HR))
        
        self.curr_ker_space.axis('off')
        self.ideal_ker_space.axis('off')
        self.real_LR_space.axis('off')
        self.real_HR_space.axis('off')
        self.fake_HR_space.axis('off')
        self.fake_LR_space.axis('off')
        self.rec_LR_space.axis('off')
        self.rec_HR_space.axis('off')
        
            
        # These line are needed in order to see the graphics at real time
        self.fig.tight_layout()
        self.fig.canvas.draw()

        title_str = f"Target_{os.path.basename(self.conf.input_image_path).split('.')[0]}_" \
                    f"iter_{self.iter}"

        date_time = datetime.now()
        str_date_time = date_time.strftime("%Y%d%m-%H%M%S")

        # if self.iter == self.conf.num_iters or self.iter == self.conf.num_iters // 3:
        if self.iter == self.conf.num_iters:
            pass
            # if not os.path.isdir("debug_img"):
            #     os.makedirs("debug_img")
            # plt.savefig(f"debug_img/{title_str}_{str_date_time}.png")
            # plt.savefig(os.path.join(self.debug_loss_path, f"{title_str}_{self.iter}_{str_date_time}.png"))
        # plt.pause(0.01)
