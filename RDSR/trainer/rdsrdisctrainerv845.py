import torch
import os
from trainer.rdsrdisctrainerv82 import RDSRDiscTrainerV8 as RDSRDiscTrainerV82
from loss.loss import GANLoss
from networks.upsample import make_up_discriminator_net

from utils.img_utils import shave_a2b, tensor2im, cal_y_psnr, calc_curr_k, calculate_psnr3
from utils.utils import set_requires_grad
from PIL import Image
from brisque import BRISQUE
from networks.downsample import HaarDownsampling


# remove haar wavlet
class RDSRDiscTrainerV845(RDSRDiscTrainerV82):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV845, self).__init__(conf, tb_logger, test_dataloader,
                                                 filename=filename, timestamp=timestamp, kernel=kernel)
        # 202404 add
        self.ref_rec_lr_w = None
        self.ref_rec_dr_w = None
        self.tar_lr_dr_w = None

        self.optimizer_tar_dr = None
        self.optimizer_ref_dr = None
        self.scheduler_tar_dr = None
        self.scheduler_ref_dr = None

    def inference_ref_dr(self, ref_ind):
        # self.en_model.train()
        self.en_model.eval()
        with torch.no_grad():
            self.ref_rec_lr_w = self.dn_model(self.ref_hr_w[ref_ind])
            # ref_rec_dr_w, _, _ = self.en_model(self.ref_rec_lr_w, self.ref_rec_lr_w)
            # tar_lr_dr_w, _, _ = self.en_model(self.tar_lr_w, self.tar_lr_w)
            ref_rec_dr_w = self.en_model(self.ref_rec_lr_w, self.ref_rec_lr_w)
            tar_lr_dr_w = self.en_model(self.tar_lr_w, self.tar_lr_w)

        ref_rec_dr_w.requires_grad = True
        tar_lr_dr_w.requires_grad = True

        self.optimizer_ref_dr = torch.optim.Adam([ref_rec_dr_w], lr=self.conf.lr_en,
                                                 betas=(self.conf.beta1, 0.999))
        self.optimizer_tar_dr = torch.optim.Adam([tar_lr_dr_w], lr=self.conf.lr_en,
                                                 betas=(self.conf.beta1, 0.999))

        self.scheduler_tar_dr = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_tar_dr, milestones=self.conf.lrs_milestone, gamma=0.5)
        self.scheduler_ref_dr = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_ref_dr, milestones=self.conf.lrs_milestone, gamma=0.5)

        self.ref_rec_dr_w = ref_rec_dr_w
        self.tar_lr_dr_w = tar_lr_dr_w

    def train_upsample(self, target_dr, target_hr_base, sr=False, en=True, matrix=False):
        set_requires_grad(self.up_disc_model, False)
        self.iter_step()
        self.sr_iter_step()
        self.dn_model.train()
        self.en_model.train()
        self.sr_model.train()
        if not self.conf.dn_freeze:
            self.optimizer_Dn.zero_grad()
        if en:
            self.optimizer_En.zero_grad()
        if sr:
            self.optimizer_Up.zero_grad()

        ref_rec_lr = self.dn_model(self.ref_hr)
        # ref_rec_lr_r = ref_rec_lr[:, 0, ...].unsqueeze(1)
        # ref_rec_lr_g = ref_rec_lr[:, 1, ...].unsqueeze(1)
        # ref_rec_lr_b = ref_rec_lr[:, 2, ...].unsqueeze(1)
        # ref_rec_lr_y = 0.299 * ref_rec_lr_r + 0.587 * ref_rec_lr_g + 0.114 * ref_rec_lr_b

        # ref_rec_lr_haar = self.haar.forward(ref_rec_lr_y)
        # ref_rec_lr_haar_hf = ref_rec_lr_haar.squeeze(2)[:, 1:, ...]

        # 20240409
        # ref_rec_dr, _, _ = self.en_model(ref_rec_lr_haar_hf, ref_rec_lr_haar_hf)
        # ref_rec_hr = self.sr_model(ref_rec_lr, ref_rec_dr)        
        ref_rec_dr_w_unit = (self.ref_rec_dr_w, )
        ref_rec_dr_w_tuple = ref_rec_dr_w_unit
        for _ in range(self.conf.batch_size - 1):
            ref_rec_dr_w_tuple += ref_rec_dr_w_unit
        ref_rec_dr_w = torch.stack(ref_rec_dr_w_tuple, dim=1).squeeze(0)
        ref_rec_hr = self.sr_model(ref_rec_lr, ref_rec_dr_w)

        # 20240409
        # tar_lr_dr, _, _ = self.en_model(self.tar_lr, self.tar_lr)
        # self.tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)
        tar_lr_dr_w_unit = (self.tar_lr_dr_w, )
        tar_lr_dr_w_tuple = tar_lr_dr_w_unit
        for _ in range(self.conf.batch_size - 1):
            tar_lr_dr_w_tuple += tar_lr_dr_w_unit
        tar_lr_dr_w = torch.stack(tar_lr_dr_w_tuple, dim=1).squeeze(0)
        self.tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr_w)

        tar_lr_rec = self.dn_model(target_hr_base)

        # 20240409
        # loss_dr = self.l1_loss(ref_rec_dr, target_dr)
        loss_dr = self.l1_loss(tar_lr_dr_w, ref_rec_dr_w)
        
        loss_ref = self.l1_loss(ref_rec_hr, shave_a2b(self.ref_hr, ref_rec_hr))

        total_loss = loss_ref * self.conf.ref_lambda
        total_loss += loss_dr * self.conf.dr_lambda
        loss_tar_sr = self.l1_loss(self.tar_hr_rec, target_hr_base)

        loss_tar_lr = self.l1_loss(tar_lr_rec, shave_a2b(self.tar_lr, tar_lr_rec))

        loss_gan = 0
        if self.conf.gan_lambda != 0:
            loss_gan = self.gan_loss.forward(self.up_disc_model(self.tar_hr_rec), True)
            total_loss += loss_gan * self.conf.gan_lambda

        loss_tar_vgg = 0
        if self.conf.vgg_tar_lambda != 0:
            loss_tar_vgg = self.vgg_loss.forward(shave_a2b(self.tar_lr, tar_lr_rec), tar_lr_rec)
            total_loss += loss_tar_vgg * self.conf.vgg_tar_lambda

        loss_ref_vgg = 0
        if self.conf.vgg_ref_lambda != 0:
            loss_ref_vgg = self.vgg_loss.forward(shave_a2b(self.ref_hr, ref_rec_hr), ref_rec_hr)
            total_loss += loss_ref_vgg * self.conf.vgg_ref_lambda

        # The purpose is for bounding baseline result
        if self.sr_iter < self.conf.target_thres:
            total_loss += loss_tar_sr * self.conf.target_lambda
            total_loss += loss_tar_lr

        loss_interpo = 0
        if self.conf.interpo_lambda != 0:
            loss_interpo = self.interpo_loss.forward(self.tar_lr, self.tar_hr_rec)
            total_loss += loss_interpo * self.conf.interpo_lambda

        loss_tv = 0
        if self.conf.tv_lambda != 0:
            loss_tv = self.tv_loss.forward(self.tar_hr_rec)
            total_loss += loss_tv * self.conf.tv_lambda

        loss_color = 0
        if self.conf.color_lambda != 0:
            loss_color = self.color_loss.forward(self.tar_lr, self.tar_hr_rec)
            total_loss += loss_color * self.conf.color_lambda

        # Add high frequency loss
        loss_hf = 0
        if self.conf.hf_lambda != 0:
            loss_hf = self.hf_loss.forward(shave_a2b(self.ref_hr, ref_rec_hr), ref_rec_hr)
            # loss_hf2 = self.hf_loss2.forward(shave_a2b(self.ref_hr, ref_rec_hr), ref_rec_hr)
            total_loss += loss_hf * self.conf.hf_lambda

        loss_ref_gv = 0
        if self.conf.gv_ref_lambda != 0:
            loss_ref_gv = self.GV_loss.forward(shave_a2b(self.ref_hr, ref_rec_hr), ref_rec_hr)
            total_loss += loss_ref_gv * self.conf.gv_ref_lambda

        if self.iter % self.conf.scale_iters == 0:
            self.logger.info(f'SR Total Loss: {self.iter}, total_loss: {total_loss}')

        total_loss.backward()
        self.update_learner(sr=sr, en=en, matrix=matrix)

        # 20240409
        # self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr, self.tar_hr_rec, tar_lr_rec)
        self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr_w, self.tar_hr_rec, tar_lr_rec)
        
        if self.iter % self.conf.evaluate_iters == 0:
            self.cal_whole_image_loss(is_dn=False)
            self.show_learning_rate()

    def update_learner(self, sr=False, en=True, dn=False, matrix=False, loss_dr=None, loss_ref=None, losses=None):
        if not self.conf.dn_freeze and dn:
            self.optimizer_Dn.step()
        # if en:
        #     self.optimizer_En.step()
        if sr:
            self.optimizer_Up.step()

        # en_scheduler = self.scheduler_En
        sr_scheduler = self.scheduler_Up

        if not self.conf.dn_freeze and dn:
            self.update_dn_lrs()

        # 20240409
        self.optimizer_ref_dr.step()
        self.optimizer_tar_dr.step()
        self.update_lrs(self.optimizer_ref_dr, self.scheduler_ref_dr, name='En')
        self.update_lrs(self.optimizer_tar_dr, self.scheduler_tar_dr, name='En')

        if matrix:
            # en_scheduler = self.matrix_scheduler_En
            sr_scheduler = self.matrix_scheduler_Up
            # if en:
            #     self.update_lrs(self.optimizer_En, en_scheduler, name='En', matrix=loss_dr)
            if sr:
                self.update_lrs(self.optimizer_Up, sr_scheduler, name='Up', matrix=loss_ref)
        else:
            # if en:
            #     self.update_lrs(self.optimizer_En, en_scheduler, name='En')
            if sr:
                self.update_lrs(self.optimizer_Up, sr_scheduler, name='Up')

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        # self.save_whole_image()
        with (torch.no_grad()):
            loss_tar_lr = self.l1_loss(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
            loss_tar_lr_vgg = self.vgg_loss.forward(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

            # calculate down sample regularization loss
            self.curr_k = calc_curr_k(self.dn_model.parameters())
            loss_gt_tar_lr = self.l1_loss(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))

            if is_dn:
                total_loss = loss_tar_lr + loss_tar_lr_vgg
                self.loss_logger.info(
                    f'{self.iter}-whole, {format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                    f'{format(loss_gt_tar_lr, ".5f")}, '
                    f'{format(total_loss, ".5f")}')

                if self.iter > self.conf.best_thres and total_loss < self.best_dn_loss:
                    self.logger.info(f'Find Better Down Sample Loss at {self.iter}, total_loss: {total_loss}')
                    self.best_dn_loss = total_loss
                    self.min_loss_dn_psnr = self.tar_lr_psnr
                    self.min_loss_dn_iter = self.iter
                    self.save_model(best=True, dn_model=True)

                if max(self.lr_psnr_list) == self.tar_lr_psnr:
                    self.max_psnr_dn_iter = self.iter
            else:
                # calculate no reference matrix
                brisque_score = self.brisque_metric.score(tensor2im(self.tar_hr_rec_w))

                loss_tar_sr = self.l1_loss(self.target_baseline_hr, self.tar_hr_rec_w)
                loss_tar_sr_vgg = self.vgg_loss.forward(self.tar_hr_rec_w,
                                                        shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w))

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
                                      f'{format(loss_tar_lr_vgg, ".5f")}, {format(loss_tar_sr2, ".5f")}, '
                                      f'{format(self.loss_disc, ".5f")}, '
                                      f'{format(brisque_score, ".5f")}, {format(total_loss, ".5f")}')

                tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))

                if self.sr_iter == 0:
                    self.base_target_loss = target_lr_loss
                    # self.logger.info(f'base_target_loss: {self.base_target_loss}')
                    self.base_target2_loss = loss_tar_sr2
                    self.logger.info(f'base_target_loss: {self.base_target_loss}, {self.base_target2_loss}')

                    # set baseline model at the first SR iter
                    self.logger.info(f'Set Base Model at {self.iter}, total_loss: {total_loss}')
                    self.save_model(best=True, dn_model=False, name='base')
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_base_w.png'))
                elif loss_tar_lr < self.base_target_loss and total_loss < self.best_sr_loss and \
                        loss_tar_sr2 < self.base_target2_loss and brisque_score < self.brisque_baseline:
                    self.best_sr_loss = total_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))

                if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                    self.max_psnr_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

                # self.save_whole_image(is_dn=False)

    def dn_evaluation(self, is_dn=True, is_first=False):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # 240408
            # # get dr
            if is_dn or is_first:
                self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
                # # inference SR
                self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)
            else:
                self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr_w)

            torch.cuda.empty_cache()

            # DownSample with SR result
            self.tar_rec_lr_w = self.dn_model(self.tar_hr_rec_w)
            # DownSample with GT
            self.tar_gt_rec_lr_w = self.dn_model(self.tar_hr_gt_w)

            # dual cycle training
            tar_rec_lr_dr = self.en_model(self.tar_rec_lr_w, self.tar_rec_lr_w)
            self.tar_hr_rec2_w = self.sr_model(self.tar_rec_lr_w, tar_rec_lr_dr)

        self.tar_hr_psnr = cal_y_psnr(tensor2im(self.tar_hr_rec_w), tensor2im(self.tar_hr_gt_w), self.conf.scale)
        self.tar_lr_psnr = cal_y_psnr(tensor2im(self.tar_rec_lr_w), tensor2im(shave_a2b(self.tar_lr_w, self.tar_rec_lr_w)), self.conf.scale)
        self.tar_gt_lr_psnr = cal_y_psnr(tensor2im(self.tar_gt_rec_lr_w), tensor2im(shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w)), self.conf.scale)

        # GT kernel PSNR
        if self.conf.kernel_gt_dir:
            if self.curr_k is None:
                self.curr_k = calc_curr_k(self.dn_model.parameters())
            self.gt_kernel_psnr = calculate_psnr3(self.curr_k, self.kernel)

        if not is_dn:
            self.sr_psnr_list.append(self.tar_hr_psnr)
        else:
            self.lr_psnr_list.append(self.tar_lr_psnr)
            self.gt_lr_psnr_list.append(self.tar_gt_lr_psnr)

    def set_baseline_img(self):
        self.baseline_model.eval()
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']
        tar_gt_w = test_data['Target_Gt']
        with torch.no_grad():
            tar_hr_rec_w = self.baseline_model(tar_lr_w)
            if self.conf.kernel_gt_dir and self.dn_gt is not None:
                tar_gt_dn_w = self.dn_gt(tar_gt_w)
        self.target_baseline_hr = tar_hr_rec_w

        # calculate baseline image PSNR
        target_baseline_psnr = cal_y_psnr(tensor2im(tar_gt_w), tensor2im(tar_hr_rec_w), self.conf.scale)
        target_dn_psnr = 0
        if self.conf.kernel_gt_dir and self.dn_gt is not None:
            target_dn_psnr = cal_y_psnr(tensor2im(tar_gt_dn_w), tensor2im(shave_a2b(tar_lr_w, tar_gt_dn_w)), self.conf.scale)
        self.baseline_psnr = target_baseline_psnr
        target_baseline_psnr = format(target_baseline_psnr, '.5f')
        target_dn_psnr = format(target_dn_psnr, '.5f')
        self.logger.info(f"Target Baseline PSNR:{target_baseline_psnr}")

        # setup no reference baseline
        self.brisque_baseline = self.brisque_metric.score(tensor2im(self.target_baseline_hr))
        tar_lr_brisque_baseline = self.brisque_metric.score(tensor2im(tar_lr_w))
        self.brisque_min = self.brisque_baseline
        self.eval_logger.info(f'{self.iter}, {target_baseline_psnr}, {target_dn_psnr}, '
                              f'{format(self.brisque_baseline, ".5f")}, {format(tar_lr_brisque_baseline, ".5f")}')