import torch
import os

from trainer.rdsrdisctrainerv42 import RDSRDiscTrainerV42
from utils.img_utils import DownScale, kernel_preprocessing, DegradationProcessing
from utils.img_utils import shave_a2b, tensor2im, calculate_psnr, calc_curr_k, calculate_psnr3
from utils.utils import set_requires_grad
from PIL import Image
from networks.upsample import make_dasr_network
from networks.downsample import make_down_double_network, make_dn_x4_k21_network, make_dn_x4_k33_network
from loss.dnloss import DownSampleRegularization
from loss.loss import CharbonnierLossV2, GANLoss
from networks.upsample import make_up_discriminator_net
from brisque import BRISQUE


class RDSRDiscTrainerV43(RDSRDiscTrainerV42):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV43, self).__init__(conf, tb_logger, test_dataloader,
                                                 filename=filename, timestamp=timestamp, kernel=kernel)

        if conf.scale == 4:
            self.dn_model = make_dn_x4_k21_network(conf).to(self.device)
            self.optimizer_Dn = torch.optim.Adam(self.dn_model.parameters(), lr=self.conf.lr_dn, betas=(self.conf.beta1, 0.999))
            self.scheduler_Dn = torch.optim.lr_scheduler.StepLR(self.optimizer_Dn, step_size=self.conf.lrs_step_size, gamma=self.conf.lrs_gamma)
            self.matrix_scheduler_Dn = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Dn, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        # self.save_whole_image()
        with (torch.no_grad()):
            loss_tar_lr = self.l1_loss(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
            loss_tar_lr_vgg = self.vgg_loss.forward(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

            # calculate down sample regularization loss
            if self.conf.kernel_gt_dir:
                self.curr_k = calc_curr_k(self.dn_model.parameters())
                loss_dn_regularization = self.dn_regularization(self.curr_k, self.target_baseline_hr, self.tar_rec_lr_w)
                loss_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
                loss_dn_kernel = loss_dn_regularization - loss_dn_bq

            loss_gt_tar_lr = self.l1_loss(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))
            if self.conf.kernel_gt_dir:
                loss_gt_dn_regularization = self.dn_regularization(self.curr_k, self.tar_hr_gt_w, self.tar_gt_rec_lr_w)
                loss_gt_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
                loss_gt_dn_kernel = loss_gt_dn_regularization - loss_gt_dn_bq

            if is_dn:
                # total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_regularization
                if self.conf.kernel_gt_dir:
                    total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_kernel
                    self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                                        f'{format(loss_dn_bq, ".5f")}, {format(loss_dn_kernel, ".5f")}, '
                                        f'{format(loss_gt_tar_lr, ".5f")}, {format(loss_gt_dn_bq, ".5f")}, {format(loss_gt_dn_kernel, ".5f")}, '
                                        f'{format(self.dn_regularization.loss_boundaries, ".5f")}, {format(self.dn_regularization.loss_sum2one, ".5f")}, '
                                        f'{format(self.dn_regularization.loss_centralized, ".5f")}, {format(self.dn_regularization.loss_sparse, ".5f")}, '
                                        f'{format(total_loss, ".5f")}')
                else:
                    total_loss = loss_tar_lr + loss_tar_lr_vgg
                    self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                                        f'{format(loss_gt_tar_lr, ".5f")}, '
                                        f'{format(total_loss, ".5f")}')

                tar_rec_lr_w_img = Image.fromarray(tensor2im(self.tar_rec_lr_w))
                if self.iter > self.conf.best_thres and total_loss < self.best_dn_loss:
                    self.logger.info(f'Find Better Down Sample Loss at {self.iter}, total_loss: {total_loss}')
                    self.best_dn_loss = total_loss
                    self.min_loss_dn_psnr = self.tar_lr_psnr
                    self.min_loss_dn_iter = self.iter
                    self.save_model(best=True, dn_model=True)
                    # self.tar_hr_rec_w = None
                    # self.tar_rec_lr_w = None

                    tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_min_loss_w.png'))

                if max(self.lr_psnr_list) == self.tar_lr_psnr:
                    self.max_psnr_dn_iter = self.iter
                    tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_max_psnr_w.png'))

                # self.save_whole_image(is_dn=True)
            else:
                # calculate no reference matrix
                brisque_score = self.brisque_metric.score(tensor2im(self.tar_hr_rec_w))

                loss_tar_sr = self.l1_loss(self.target_baseline_hr, self.tar_hr_rec_w)
                loss_tar_sr_vgg = self.vgg_loss.forward(self.tar_hr_rec_w, shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w))

                # loss_interpo = self.interpo_loss.forward(self.tar_lr_w, self.tar_hr_rec_w)
                # loss_tv = self.tv_loss.forward(self.tar_hr_rec_w)
                # loss_tar_hf = self.hf_loss.forward(shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w), self.tar_hr_rec_w)

                # add for set the baseline
                loss_tar_sr2 = self.l1_loss(shave_a2b(self.target_baseline_hr, self.tar_hr_rec2_w), self.tar_hr_rec2_w)

                target_lr_loss += loss_tar_lr
                target_lr_loss += loss_tar_lr_vgg * self.conf.vgg_lambda

                target_sr_loss += loss_tar_sr
                target_sr_loss += loss_tar_sr_vgg * self.conf.vgg_lambda

                # try lr: 3, sr: 1
                target_total_loss = target_lr_loss * self.conf.target_lr_lambda + target_sr_loss * self.conf.target_sr_lambda

                total_loss = self.conf.total_target_lambda * target_total_loss

                self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_sr, ".5f")}, '
                                      f'{format(loss_tar_sr_vgg, ".5f")}, {format(loss_tar_lr, ".5f")}, '
                                      f'{format(loss_tar_lr_vgg, ".5f")}, {format(self.loss_disc, ".5f")}, '
                                      f'{format(loss_tar_sr2, ".5f")}, {format(brisque_score, ".5f")}, '
                                      f'{format(total_loss, ".5f")}')

                tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))

                if self.sr_iter == 0:
                    self.base_target_loss = target_lr_loss
                    self.base_target2_loss = loss_tar_sr2
                    self.logger.info(f'base_target_loss: {self.base_target_loss}, {self.base_target2_loss}')

                    # set baseline model at the first SR iter
                    self.logger.info(f'Set Base Model at {self.iter}, total_loss: {total_loss}')
                    self.save_model(best=True, dn_model=False, name='base')
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_base_w.png'))
                elif loss_tar_lr < self.base_target_loss and total_loss < self.best_sr_loss and \
                        brisque_score < self.brisque_baseline:
                    self.best_sr_loss = total_loss
                    self.best_target_loss = target_lr_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))

                if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                    self.max_psnr_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

                # self.save_whole_image(is_dn=False)


class RDSRDiscTrainerV431(RDSRDiscTrainerV43):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV431, self).__init__(conf, tb_logger, test_dataloader,
                                                 filename=filename, timestamp=timestamp, kernel=kernel)

        if conf.scale == 4:
            self.dn_model = make_dn_x4_k33_network(conf).to(self.device)
            self.optimizer_Dn = torch.optim.Adam(self.dn_model.parameters(), lr=self.conf.lr_dn, betas=(self.conf.beta1, 0.999))
            self.scheduler_Dn = torch.optim.lr_scheduler.StepLR(self.optimizer_Dn, step_size=self.conf.lrs_step_size, gamma=self.conf.lrs_gamma)
            self.matrix_scheduler_Dn = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Dn, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)


class RDSRDiscTrainerV44(RDSRDiscTrainerV43):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV44, self).__init__(conf, tb_logger, test_dataloader,
                                                 filename=filename, timestamp=timestamp, kernel=kernel)

        if conf.scale == 4:
            self.dn_model = make_dn_x4_k33_network(conf).to(self.device)
            self.optimizer_Dn = torch.optim.Adam(self.dn_model.parameters(), lr=self.conf.lr_dn, betas=(self.conf.beta1, 0.999))
            self.scheduler_Dn = torch.optim.lr_scheduler.StepLR(self.optimizer_Dn, step_size=self.conf.lrs_step_size, gamma=self.conf.lrs_gamma)
            self.matrix_scheduler_Dn = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Dn, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)

    def start_train_dn(self, target_hr_base):
        self.iter_step()

        self.dn_model.train()
        self.optimizer_Dn.zero_grad()
        tar_rec_lr = self.dn_model(target_hr_base)

        # calculate L1 loss
        total_loss = 0
        dn_l1_loss = self.l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        total_loss += dn_l1_loss

        loss_tar_vgg = 0
        if self.conf.vgg_tar_lambda != 0:
            loss_tar_vgg = self.vgg_loss.forward(shave_a2b(self.tar_lr, tar_rec_lr), tar_rec_lr)
            total_loss += loss_tar_vgg * self.conf.vgg_tar_lambda

        if self.iter % self.conf.scale_iters == 0:
            # self.loss_logger.info(f'{self.iter}, {format(dn_l1_loss, ".5f")}, {format(loss_tar_vgg, ".5f")}, , ,')
            self.loss_list.append(total_loss.item())
        total_loss.backward()
        # print(total_loss)

        self.optimizer_Dn.step()

        # Update learning rate
        self.update_dn_lrs()
        # self.update_matrix_dn_lrs(total_loss)

        if self.iter % self.conf.plot_iters == 0:
            tar_img = Image.fromarray(tensor2im(self.tar_lr))
            tar_hr_img = Image.fromarray(tensor2im(target_hr_base))
            tar_rec_lr_img = Image.fromarray(tensor2im(tar_rec_lr))

            tar_img.save(os.path.join(self.save_path, 'tar_img.png'))
            tar_hr_img.save(os.path.join(self.save_path, 'tar_hr_img.png'))
            tar_rec_lr_img.save(os.path.join(self.save_path, 'tar_lr_rec.png'))
            # self.dn_model_evaluation()
            self.dn_model.eval()
            self.dn_evaluation()

        if self.iter % self.conf.evaluate_iters == 0:
            # self.dn_evaluation()
            self.eval_logger.info(f'{self.iter}, {format(self.tar_hr_psnr, ".5f")}, {format(self.tar_lr_psnr, ".5f")}, '
                                  f'{format(self.tar_gt_lr_psnr, ".5f")}, {format(self.gt_kernel_psnr, ".5f")}')

            self.cal_whole_image_loss(is_dn=True)
            self.show_learning_rate(is_dn=True)