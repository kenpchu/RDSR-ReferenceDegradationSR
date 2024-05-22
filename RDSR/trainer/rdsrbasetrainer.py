import os
import logging
import torch
import torch.nn as nn

from loss import loss
from loss.loss import VGG
from PIL import Image
from utils.img_utils import shave_a2b, tensor2im, calculate_psnr


class RDSRBaseTrainer(object):
    train_log_name = 'train_loggers'
    network_name_list = ['En', 'Up', 'Dn']

    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        self.conf = conf
        self.scale = conf.scale
        # GPU configuration settings
        self.device = torch.device('cpu' if conf.cpu else 'cuda')

        # Log Settings
        self.filename = filename
        self.tb_logger = tb_logger
        # Log Path settings
        self.timestamp = timestamp
        self.save_path = os.path.join(self.train_log_name, f'{conf.exp_name}_{timestamp}', filename)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # init iteration
        self.iter = 0
        self.sr_iter = 0

        # test dataset
        self.test_loader = test_dataloader

        # [Customization] implement kernel preprocessing
        self.kernel = None
        # if kernel is not None:
        #     self.init_kernel(kernel)
        #     # kernel_t = torch.from_numpy(kernel).float().cuda()
        #     # self.dn_by_kernel = DegradationProcessing(conf, kernel_t)
        #     pass

        # input data initialize
        self.ref_hr = None
        self.ref_bq_dn = None
        self.tar_lr = None
        self.tar_bq_up = None
        self.target_baseline_hr = None

        # evaluate values initialization
        self.eval_list = []
        self.loss_list = []
        self.lr_list = []

        self.sr_psnr_list = []
        self.lr_psnr_list = []
        self.gt_lr_psnr_list = []

        self.tmp_psnr = 0
        self.tmp_ref_psnr = 0

        self.tar_upper_bound_psnr = 0
        self.tar_sr_low_bound_psnr = 0

        self.min_loss_dn_psnr = 0
        self.best_dn_loss = 999999
        self.min_loss_sr_psnr = 0
        self.best_sr_loss = 999999
        self.best_target_loss = 999999
        self.min_loss_sr_iter = -1
        self.min_loss_dn_iter = -1
        self.max_psnr_sr_iter = -1
        self.max_psnr_dn_iter = -1
        self.baseline_psnr = -1

        self.tar_lr_psnr = 0
        self.tar_gt_lr_psnr = 0
        self.tar_hr_psnr = 0
        self.gt_kernel_psnr = 0
        self.ref_lr_psnr = 0
        self.ref_hr_psnr = 0
        self.ref_gt_hr_psnr = 0

        # TensorBoard & Logger
        self.iter = 0
        self.sr_iter = 0
        self.epoch = 0
        self.filename = filename
        self.tb_logger = tb_logger

        self.logger = None
        self.eval_logger = None
        self.loss_logger = None
        self.lr_logger = None
        self.init_loggers()

        # loss initialize
        self.tv_loss = loss.TVLoss().cuda()
        # self.interpo_loss = loss.InterpolationLoss(1 / self.scale).cuda()
        self.interpo_loss = loss.InterpolationLoss(self.scale).cuda()
        self.interpo_loss2 = loss.InterpolationLoss2(self.scale).cuda()
        self.vgg_loss = VGG('22', rgb_range=255).cuda()
        self.color_loss = loss.ColorLoss(self.scale)
        self.hf_loss = loss.HighFrequencyLoss().cuda()
        self.hf_loss2 = loss.HighFrequencyLoss2().cuda()
        self.l1_loss_ori = nn.L1Loss()
        self.l1_loss = loss.CharbonnierLoss().cuda()
        self.lr_l1_loss = loss.CharbonnierLoss().cuda()
        self.GV_loss = loss.GradientVariance(self.conf.patch_size).cuda()

        # network initialization
        # [Customization]
        self.dn_model = None
        self.dn_gt = None
        self.sr_model = None
        self.en_model = None
        self.baseline_model = None
        self.finetune_model = None

        # [Customization] optimizer initialization
        self.optimizer_Dn = None
        self.optimizer_Up = None
        self.optimizer_En = None
        # self.init_optimizer()

        # [Customization] scheduler initialization
        self.scheduler_Dn = None
        self.scheduler_En = None
        self.scheduler_Up = None

        # [Customization] advanced scheduler initialization
        self.matrix_scheduler_Dn = None
        self.matrix_scheduler_Up = None
        self.matrix_scheduler_En = None
        # self.init_scheduler()

    # initial loggers
    def init_loggers(self):
        self.logger = logging.getLogger(self.timestamp)
        self.eval_logger = logging.getLogger('eval')
        self.loss_logger = logging.getLogger('loss')
        self.lr_logger = logging.getLogger('lr')
        self.eval_logger.info(f'{self.filename},')
        self.loss_logger.info(f'{self.filename},')
        self.lr_logger.info(f'{self.filename},')

    def init_kernel(self, kernel_path):
        # [Customization] Need to implement preprocess kernel
        raise NotImplementedError

    def init_optimizer(self, is_dn=True, is_en=True, is_up=True):
        # [Customization] optimizer initialization
        assert self.dn_model is not None
        assert self.sr_model is not None
        assert self.en_model is not None
        if self.conf.optim == 'adam':
            if hasattr(self.dn_model, 'parameters') and is_dn:
                self.optimizer_Dn = torch.optim.Adam(self.dn_model.parameters(), lr=self.conf.lr_dn, betas=(self.conf.beta1, 0.999))
            if self.sr_model is not None and is_up:
                self.optimizer_Up = torch.optim.Adam(self.sr_model.parameters(), lr=self.conf.lr_up, betas=(self.conf.beta1, 0.999))
            if self.en_model is not None and is_en:
                self.optimizer_En = torch.optim.Adam(self.en_model.parameters(), lr=self.conf.lr_en, betas=(self.conf.beta1, 0.999))
        elif self.conf.optim == 'SGD':
            if hasattr(self.dn_model, 'parameters') and is_dn:
                self.optimizer_Dn = torch.optim.SGD(self.dn_model.parameters(), lr=self.conf.lr_dn, momentum=0.9)
            if self.sr_model is not None and is_up:
                self.optimizer_Up = torch.optim.SGD(self.sr_model.parameters(), lr=self.conf.lr_up, momentum=0.9)
            if self.en_model is not None and is_en:
                self.optimizer_En = torch.optim.SGD(self.en_model.parameters(), lr=self.conf.lr_en, momentum=0.9)

    def init_scheduler(self, is_dn=True, is_en=True, is_up=True):
        if hasattr(self.dn_model, 'parameters') and is_dn:
            self.scheduler_Dn = torch.optim.lr_scheduler.StepLR(self.optimizer_Dn, step_size=self.conf.lrs_step_size, gamma=self.conf.lrs_gamma)
            self.matrix_scheduler_Dn = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Dn, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)
        if self.sr_model is not None and is_up:
            self.scheduler_Up = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_Up, milestones=self.conf.lrs_milestone, gamma=0.5)
            self.matrix_scheduler_Up = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Up, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)

        if self.en_model is not None and is_en:
            self.scheduler_En = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_En, milestones=self.conf.lrs_milestone, gamma=0.5)
            self.matrix_scheduler_En = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_En, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)

    def flush_log(self):
        for handle in self.logger.handlers:
            handle.flush()
        for handle in self.eval_logger.handlers:
            handle.flush()
        for handle in self.loss_logger.handlers:
            handle.flush()
        for handle in self.lr_logger.handlers:
            handle.flush()

    # set input image & baseline image
    def set_input(self, data_dict):
        self.ref_hr = data_dict['Ref_HR']
        self.ref_bq_dn = data_dict['Ref_bq_dn']
        self.tar_lr = data_dict['Tar_LR']
        self.tar_bq_up = data_dict['Tar_bq_up']

    def set_dn_input(self, data_dict):
        self.tar_lr = data_dict['Tar_LR']
        self.tar_bq_up = data_dict['Tar_bq_up']

    def set_baseline_img(self):
        self.baseline_model.eval()
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']
        tar_gt_w = test_data['Target_Gt']
        with torch.no_grad():
            # dr = self.baseline_model.E(tar_lr_w, tar_lr_w)
            # print(dr)
            tar_hr_rec_w = self.baseline_model(tar_lr_w)
            if self.conf.kernel_gt_dir and self.dn_gt is not None:
                tar_gt_dn_w = self.dn_gt(tar_gt_w)
        self.target_baseline_hr = tar_hr_rec_w

        # calculate baseline image PSNR
        target_baseline_psnr = calculate_psnr(tar_gt_w, tar_hr_rec_w)
        target_dn_psnr = 0
        if self.conf.kernel_gt_dir and self.dn_gt is not None:
            target_dn_psnr = calculate_psnr(tar_gt_dn_w, shave_a2b(tar_lr_w, tar_gt_dn_w))
        self.baseline_psnr = target_baseline_psnr
        target_baseline_psnr = format(target_baseline_psnr, '.5f')
        target_dn_psnr = format(target_dn_psnr, '.5f')
        self.logger.info(f"Target Baseline PSNR:{target_baseline_psnr}")
        self.eval_logger.info(f"{self.iter}, {target_baseline_psnr}, {target_dn_psnr}")

        # save baseline image
        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'target_hr_baseline_w.png'))

        tar_lr_w_path = os.path.join(self.save_path, 'target_lr_w.png')
        tar_lr_w_img = Image.fromarray(tensor2im(tar_lr_w))
        tar_lr_w_img.save(tar_lr_w_path)

        tar_hr_gt_w_path = os.path.join(self.save_path, 'target_hr_gt_w.png')
        tar_hr_gt_w_img = Image.fromarray(tensor2im(tar_gt_w))
        tar_hr_gt_w_img.save(tar_hr_gt_w_path)

    # set target images
    def get_target_baseline_result(self):
        self.baseline_model.eval()
        # self.baseline_model.E.training = False
        with torch.no_grad():
            tar_hr_rec = self.baseline_model(self.tar_lr)
        return tar_hr_rec

    def get_target_baseline_representation(self):
        # [Customization] optimizer initialization
        raise NotImplementedError

    # load model functions
    def load_sr_model(self, path):
        self.finetune_model.load_state_dict(torch.load(path), strict=False)

    def load_dn_model(self, path):
        self.dn_model.load_state_dict(torch.load(path), strict=False)

    def load_en_model(self, path):
        self.en_model.load_state_dict(torch.load(path), strict=False)

    def load_baseline_model(self, path):
        self.baseline_model.load_state_dict(torch.load(path), strict=False)

    def load_pretrain_model(self, path):
        self.baseline_model.load_state_dict(torch.load(path), strict=False)
        self.finetune_model.load_state_dict(torch.load(path), strict=False)

    @staticmethod
    def freeze_network(model):
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_network(model):
        model.train()
        for name, param in model.named_parameters():
            param.requires_grad = True

    def freeze_sr_network(self):
        self.en_model.eval()
        self.sr_model.eval()
        for name, param in self.en_model.named_parameters():
            param.requires_grad = False
        for name, param in self.sr_model.named_parameters():
            param.requires_grad = False

    # def freeze_sr_network(self):
    #     self.sr_model.eval()
    #     for name, param in self.sr_model.G.named_parameters():
    #         param.requires_grad = False
    #
    #     for name, param in self.sr_model.E.named_parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_encoder_network(self):
    #     self.en_model.train()
    #     for name, param in self.en_model.named_parameters():
    #         param.requires_grad = True
    #
    # def unfreeze_sr_network(self):
    #     self.sr_model.train()
    #     for name, param in self.sr_model.G.named_parameters():
    #         param.requires_grad = True

    def iter_step(self, cnt=0):
        if not cnt:
            self.iter += 1
            if self.iter % self.conf.evaluate_iters == 0:
                self.logger.info(f'iteration: {self.iter}')
        else:
            self.iter = cnt

    def sr_iter_step(self, cnt=0):
        if not cnt:
            self.sr_iter += 1
            if self.sr_iter % self.conf.evaluate_iters == 0:
                self.logger.info(f'sr iteration: {self.iter}')
        else:
            self.sr_iter = cnt

    # evaluation functions
    def plot_eval(self, ref_rec_lr, ref_rec_hr, ref_dr, tar_hr_rec, tar_lr_rec):
        if self.iter % self.conf.plot_iters == 0:
            ref_img = Image.fromarray(tensor2im(self.ref_hr))
            ref_rec_lr_img = Image.fromarray(tensor2im(ref_rec_lr))
            ref_rec_hr_img = Image.fromarray(tensor2im(ref_rec_hr))

            ref_img.save(os.path.join(self.save_path, 'ref_img.png'))
            ref_rec_lr_img.save(os.path.join(self.save_path, 'ref_rec_lr_img.png'))
            ref_rec_hr_img.save(os.path.join(self.save_path, 'ref_rec_hr_img.png'))

        if self.iter % self.conf.evaluate_iters == 0:
            if hasattr(self.dn_model, 'parameters'):
                self.dn_model.eval()
            self.en_model.eval()
            self.sr_model.eval()
            self.dn_evaluation(is_dn=False)
            self.sr_evaluation()

            self.eval_logger.info(f'{self.iter}, {format(self.tar_hr_psnr, ".5f")}, {format(self.tar_lr_psnr, ".5f")}, '
                                  f'{format(self.ref_hr_psnr, ".5f")}, {format(self.ref_lr_psnr, ".5f")}, '
                                  f'{format(self.tar_gt_lr_psnr, ".5f")}, {format(self.ref_gt_hr_psnr, ".5f")}')

    # TODO: replace sr_evaluation, dn_evaluation, ref_whole_loss, target_whole_loss
    def sr_evaluation(self, is_ref_loss=True):
        test_data = next(iter(self.test_loader))
        ref_hr_w = test_data['Ref_Img']

        # TODO: Need to implement all ref images to evaluate performance
        ref_lr_gt_w = None
        if 'Ref_Gt' in test_data.keys():
            ref_lr_gt_w = test_data['Ref_Gt']

        with torch.no_grad():
            ref_lr_rec_w = self.dn_model(ref_hr_w)
            ref_dr = self.en_model(ref_lr_rec_w, ref_lr_rec_w)
            ref_hr_rec_w = self.sr_model(ref_lr_rec_w, ref_dr)
            # almost no use
            if ref_lr_gt_w is not None:
                ref_gt_hr_rec_w = self.sr_model(ref_lr_gt_w)

        if is_ref_loss:
            self.ref_whole_loss(ref_hr_w, ref_lr_rec_w, ref_hr_rec_w)

        self.ref_hr_psnr = calculate_psnr(ref_hr_rec_w, shave_a2b(ref_hr_w, ref_hr_rec_w))

        # almost no use
        if ref_lr_gt_w is not None:
            self.ref_lr_psnr = calculate_psnr(ref_lr_rec_w, shave_a2b(ref_lr_gt_w, ref_lr_rec_w))
            self.ref_gt_hr_psnr = calculate_psnr(ref_gt_hr_rec_w, shave_a2b(ref_hr_w, ref_gt_hr_rec_w))

        ref_hr_rec_w_img = Image.fromarray(tensor2im(ref_hr_rec_w))
        ref_hr_rec_w_img.save(os.path.join(self.save_path, 'ref_hr_rec_w.png'))

        ref_lr_rec_w_img = Image.fromarray(tensor2im(ref_lr_rec_w))
        ref_lr_rec_w_img.save(os.path.join(self.save_path, 'ref_lr_rec_w.png'))

        return self.ref_hr_psnr

    def ref_whole_loss(self, ref_hr_w, ref_lr_rec_w, ref_hr_rec_w):
        with torch.no_grad():
            total_loss = 0
            loss_ref = self.l1_loss(shave_a2b(ref_hr_w, ref_hr_rec_w), ref_hr_rec_w)

            total_loss += loss_ref * self.conf.ref_lambda

            loss_ref_vgg = self.vgg_loss.forward(shave_a2b(ref_hr_w, ref_hr_rec_w), ref_hr_rec_w)
            total_loss += loss_ref_vgg * self.conf.vgg_ref_lambda

            loss_hf = self.hf_loss.forward(shave_a2b(ref_hr_w, ref_hr_rec_w), ref_hr_rec_w)
            total_loss += loss_hf * self.conf.hf_lambda

            loss_gv = self.GV_loss.forward(shave_a2b(ref_hr_w, ref_hr_rec_w), ref_hr_rec_w)
            total_loss += loss_gv * self.conf.gv_ref_lambda

            # TODO: ref hr & ref lr upsample VGG
            # loss_ref_vgg = self.vgg_loss.forward(ref_hr_w, ref_hr_rec_w)
            # total_loss += loss_ref_vgg * self.conf.vgg_ref_lambda

            self.loss_logger.info(f'{self.iter}-ref-whole, {format(loss_ref, ".5f")}, {format(loss_ref_vgg, ".5f")}, '
                                  f'{format(loss_hf, ".5f")}, {format(loss_gv, ".5f")}, {format(total_loss, ".5f")}')

    def dn_evaluation(self, is_dn=True):
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']
        tar_hr_gt_w = test_data['Target_Gt']

        with torch.no_grad():
            # get dr
            tar_lr_dn = self.en_model(tar_lr_w, tar_lr_w)
            # print(tar_lr_dn)
            # inference SR
            tar_hr_rec_w = self.sr_model(tar_lr_w, tar_lr_dn)
            # DownSample with SR result
            tar_rec_lr_w = self.dn_model(tar_hr_rec_w)
            # DownSample with GT
            tar_gt_rec_lr_w = self.dn_model(tar_hr_gt_w)

        is_best_img = self.target_whole_loss(tar_lr_w, tar_hr_rec_w, tar_rec_lr_w, is_dn=is_dn)

        self.tar_hr_psnr = calculate_psnr(tar_hr_rec_w, tar_hr_gt_w)
        self.tar_lr_psnr = calculate_psnr(tar_rec_lr_w, shave_a2b(tar_lr_w, tar_rec_lr_w))
        self.tar_gt_lr_psnr = calculate_psnr(tar_gt_rec_lr_w, shave_a2b(tar_lr_w, tar_gt_rec_lr_w))

        # self.tar_lr_psnr_cmp = calculate_psnr(shave_a2b(tar_rec_lr_w_cmp, tar_rec_lr_w), shave_a2b(tar_lr_w, tar_rec_lr_w))
        # self.tar_gt_lr_psnr_cmp = calculate_psnr(shave_a2b(tar_gt_rec_lr_w_cmp, tar_rec_lr_w), shave_a2b(tar_lr_w, tar_gt_rec_lr_w))

        if not is_dn:
            self.sr_psnr_list.append(self.tar_hr_psnr)
        else:
            self.lr_psnr_list.append(self.tar_lr_psnr)
            self.gt_lr_psnr_list.append(self.tar_gt_lr_psnr)

        if self.iter % self.conf.evaluate_iters == 0:
            tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w))
            tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_w.png'))

            if not is_dn and is_best_img:
                tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))
                # TODO: might be a bug, fixed it on 0806
                # if self.tar_hr_psnr > self.min_loss_sr_psnr:
                #     self.min_loss_sr_psnr = self.tar_hr_psnr
                #     self.min_loss_sr_iter = self.iter
                self.min_loss_sr_psnr = self.tar_hr_psnr
                self.min_loss_sr_iter = self.iter
            if not is_dn and max(self.sr_psnr_list) == self.tar_hr_psnr:
                self.max_psnr_sr_iter = self.iter
                tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

            tar_rec_lr_w_img = Image.fromarray(tensor2im(tar_rec_lr_w))
            tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_w.png'))

            if is_dn and is_best_img:
                tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_min_loss_w.png'))
                # TODO: might be a bug, fixed it on 0806
                # if self.tar_lr_psnr > self.min_loss_dn_psnr:
                #     self.min_loss_dn_psnr = self.tar_lr_psnr
                #     self.min_loss_dn_iter = self.iter
                self.min_loss_dn_psnr = self.tar_lr_psnr
                self.min_loss_dn_iter = self.iter
            if is_dn and max(self.lr_psnr_list) == self.tar_lr_psnr:
                self.max_psnr_dn_iter = self.iter
                tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_max_psnr_w.png'))

            tar_gt_rec_lr_w_img = Image.fromarray(tensor2im(tar_gt_rec_lr_w))
            tar_gt_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_gt_rec_lr_w.png'))

        return self.tar_lr_psnr

    def target_whole_loss(self, tar_lr_w, tar_hr_rec_w, tar_rec_lr_w, is_dn=True):
        # This function is observed for overall target image quality
        is_best_img = False

        with torch.no_grad():
            total_loss = 0
            loss_tar_sr = self.l1_loss(self.target_baseline_hr, tar_hr_rec_w)
            loss_tar_lr = self.l1_loss(tar_rec_lr_w, shave_a2b(tar_lr_w, tar_rec_lr_w))

            total_loss += loss_tar_sr * self.conf.target_lambda
            total_loss += loss_tar_lr

            loss_tar_vgg = 0
            if self.conf.vgg_tar_lambda != 0:
                loss_tar_vgg = self.vgg_loss.forward(tar_rec_lr_w, shave_a2b(tar_lr_w, tar_rec_lr_w))
                total_loss += loss_tar_vgg * self.conf.vgg_tar_lambda
                # TODO: target lr unsample & target hr VGG

            loss_tar_sr_vgg = self.vgg_loss.forward(tar_hr_rec_w, shave_a2b(self.target_baseline_hr, tar_hr_rec_w))
            total_loss += loss_tar_sr_vgg * self.conf.vgg_tar_lambda

            loss_interpo = 0
            if self.conf.interpo_lambda != 0:
                loss_interpo = self.interpo_loss.forward(tar_lr_w, tar_hr_rec_w)
                total_loss += loss_interpo

            loss_tv = 0
            if self.conf.tv_lambda != 0:
                loss_tv = self.tv_loss.forward(tar_hr_rec_w)
                total_loss += loss_tv * self.conf.tv_lambda

            loss_tar_hf = 0
            if self.conf.hf_lambda != 0:
                loss_tar_hf = self.hf_loss.forward(shave_a2b(self.target_baseline_hr, tar_hr_rec_w), tar_hr_rec_w)
                total_loss += loss_tar_hf * self.conf.hf_lambda

            self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_sr, ".5f")}, '
                                  f'{format(loss_tar_sr_vgg, ".5f")}, {format(loss_tar_hf, ".5f")}, '
                                  f'{format(loss_interpo, ".5f")}, {format(loss_tv, ".5f")}, '
                                  f'{format(loss_tar_vgg, ".5f")}, {format(loss_tar_lr, ".5f")}, '
                                  f'{format(total_loss, ".5f")}')

            if self.iter > self.conf.best_thres and total_loss < self.best_sr_loss:
                self.best_sr_loss = total_loss
                self.save_model(best=True, dn_model=False)
                self.save_best_img()
                self.logger.info(f'Find Better Loss at {self.iter}, total_loss: {total_loss}')
                is_best_img = True

        return is_best_img

    # learning rate functions
    def update_dn_lrs(self):
        self.scheduler_Dn.step()
        current_lr = self.optimizer_Dn.param_groups[0]['lr']
        if self.iter % self.conf.evaluate_iters == 0:
            # self.lr_logger.info(f'{self.iter}, , ,{current_lr}')
            self.lr_list.append(current_lr)

    def update_matrix_dn_lrs(self, matrix):
        self.matrix_scheduler_Dn.step(matrix)
        current_lr = self.optimizer_Dn.param_groups[0]['lr']
        # if self.iter % self.conf.scale_iters == 0:
        if self.iter % self.conf.evaluate_iters == 0:
            # self.lr_logger.info(f'{self.iter}, , ,{current_lr}')
            self.lr_list.append(current_lr)

    def update_lrs(self, optimizer, scheduler, name='', matrix=''):
        idx = -1
        if name in self.network_name_list:
            idx = self.network_name_list.index(name)
        # More General
        if matrix:
            scheduler.step(matrix)
        else:
            scheduler.step()

        curr_lr = optimizer.param_groups[0]['lr']
        # if self.iter % self.conf.evaluate_iters == 0:
        #     if idx == 0:
        #         self.lr_logger.info(f'{self.iter},{curr_lr},,')
        #     elif idx == 1:
        #         self.lr_logger.info(f'{self.iter},,{curr_lr},')
        self.lr_list.append(curr_lr)

    def show_learning_rate(self, is_dn=False):
        curr_dn_lr = 0
        if hasattr(self.dn_model, 'parameters'):
            curr_dn_lr = self.optimizer_Dn.param_groups[0]['lr']
        if not is_dn:
            curr_en_lr = self.optimizer_En.param_groups[0]['lr']
            curr_up_lr = self.optimizer_Up.param_groups[0]['lr']
            self.lr_logger.info(f'{self.iter},{curr_dn_lr},{curr_en_lr},{curr_up_lr}')
        else:
            self.lr_logger.info(f'{self.iter},{curr_dn_lr}')

    # save model & images
    def save_model(self, best=False, dn_model=True):
        if self.timestamp:
            output_path = os.path.join(self.conf.train_log, f'{self.conf.exp_name}_{self.timestamp}', self.filename)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.save_path = output_path
        if not best:
            dn_model_path = os.path.join(self.save_path, f'model_dn_{self.filename}_{self.iter}.pt')
            sr_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_{self.iter}.pt')
        else:
            dn_model_path = os.path.join(self.save_path, f'model_dn_{self.filename}_best.pt')
            sr_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_best.pt')
            if dn_model:
                torch.save(self.dn_model.state_dict(), dn_model_path)
            else:
                # torch.save(self.sr_model.state_dict(), sr_model_path)
                torch.save(self.finetune_model.state_dict(), sr_model_path)
            return
        if hasattr(self.dn_model, 'parameters'):
            torch.save(self.dn_model.state_dict(), dn_model_path)
        # torch.save(self.sr_model, sr_model_path)
        torch.save(self.finetune_model.state_dict(), sr_model_path)

    def save_best_img(self):
        # test_data = self.test_dataloader.__getitem__(0)
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']

        with torch.no_grad():
            tar_lr_dr_w = self.en_model(tar_lr_w, tar_lr_w)
            tar_hr_rec_w = self.sr_model(tar_lr_w, tar_lr_dr_w)

        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_w_best.png'))

    def save_img(self, img, img_name):
        save_path = os.path.join(self.save_path, img_name + '.png')
        save_patch_img = Image.fromarray(tensor2im(img))
        save_patch_img.save(save_path)

    def finish(self):
        # save_model
        if self.conf.save_model:
            self.save_model()
        if self.conf.save_image:
            pass
        self.eval_logger.info(f'min_loss_sr_psnr, max_sr_psnr,  dasr_psnr, min_loss_diff_dasr, '
                              f'max_psnr_diff_dasr, min_loss_sr_iter, max_psnr_sr_iter, '
                              f'min_loss_lr_iter, max_psnr_lr_iter')
        min_loss_diff_dasr = format(self.min_loss_sr_psnr - self.baseline_psnr, ".5f")
        max_psnr_diff_dasr = format(max(self.sr_psnr_list) - self.baseline_psnr, ".5f")
        min_loss_sr_psnr = format(self.min_loss_sr_psnr, ".5f")
        max_sr_psnr = format(max(self.sr_psnr_list), ".5f")
        dasr_psnr = format(self.baseline_psnr, ".5f")
        min_loss_dn_psnr = format(self.min_loss_dn_psnr, ".5f")

        max_lr_psnr = -1
        if len(self.lr_psnr_list) > 0:
            max_lr_psnr = format(max(self.lr_psnr_list), ".5f")
        # max_gt_lr_psnr = format(max(self.gt_lr_psnr_list), ".5f")
        self.eval_logger.info(f'{min_loss_sr_psnr}, {max_sr_psnr}, {dasr_psnr}, {min_loss_diff_dasr}, '
                              f'{max_psnr_diff_dasr}, {self.min_loss_sr_iter}, {self.max_psnr_sr_iter}, '
                              f'{min_loss_dn_psnr}, {max_lr_psnr}, {self.min_loss_dn_iter}, {self.max_psnr_dn_iter}')

        self.tb_logger.flush()
        print('*' * 60 + '\n')

    def start_train_dn(self):
        # [Customization]
        raise NotImplementedError
        pass

    def start_train_rdsr(self, target_dr, target_hr_baseline, sr=False, en=True, matrix=False):
        # [Customization]
        raise NotImplementedError
        pass



