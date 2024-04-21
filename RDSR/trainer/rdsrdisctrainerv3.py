import torch
import os
from trainer.rdsrdisctrainerv2 import RDSRDiscTrainerV2
from loss.loss import GANLoss
from networks.upsample import make_up_discriminator_net

from utils.img_utils import shave_a2b, tensor2im, calculate_psnr, calc_curr_k, calculate_psnr3
from utils.utils import set_requires_grad
from PIL import Image
from networks.downsample import HaarDownsampling


class RDSRDiscTrainerV3(RDSRDiscTrainerV2):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV3, self).__init__(conf, tb_logger, test_dataloader,
                                              filename=filename, timestamp=timestamp, kernel=kernel)
        self.haar = HaarDownsampling(1)

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
        ref_rec_lr_r = ref_rec_lr[:, 0, ...].unsqueeze(1)
        ref_rec_lr_g = ref_rec_lr[:, 1, ...].unsqueeze(1)
        ref_rec_lr_b = ref_rec_lr[:, 2, ...].unsqueeze(1)
        ref_rec_lr_y = 0.299 * ref_rec_lr_r + 0.587 * ref_rec_lr_g + 0.114 * ref_rec_lr_b

        ref_rec_lr_haar = self.haar.forward(ref_rec_lr_y)
        ref_rec_lr_haar_hf = ref_rec_lr_haar.squeeze(2)[:, 1:, ...]
        # ref_rec_dr, _, _ = self.en_model(ref_rec_lr, ref_rec_lr)
        ref_rec_dr, _, _ = self.en_model(ref_rec_lr_haar_hf, ref_rec_lr_haar_hf)
        ref_rec_hr = self.sr_model(ref_rec_lr, ref_rec_dr)

        tar_lr_dr, _, _ = self.en_model(self.tar_lr, self.tar_lr)
        self.tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)
        tar_lr_rec = self.dn_model(target_hr_base)

        loss_dr = self.l1_loss(ref_rec_dr, target_dr)
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

        self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr, self.tar_hr_rec, tar_lr_rec)
        if self.iter % self.conf.evaluate_iters == 0:
            self.cal_whole_image_loss(is_dn=False)
            self.show_learning_rate()
