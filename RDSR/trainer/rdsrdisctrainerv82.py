import torch
import os
from trainer.rdsrdisctrainerv3 import RDSRDiscTrainerV3
from loss.loss import GANLoss
from networks.upsample import make_up_discriminator_net

from utils.img_utils import shave_a2b, tensor2im, calculate_psnr, calc_curr_k, calculate_psnr3
from utils.utils import set_requires_grad
from PIL import Image
from brisque import BRISQUE
from networks.downsample import HaarDownsampling

import matplotlib.pyplot as plt
# remove haar wavlet
class RDSRDiscTrainerV8(RDSRDiscTrainerV3):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV8, self).__init__(conf, tb_logger, test_dataloader,
                                              filename=filename, timestamp=timestamp, kernel=kernel)
        self.tar_hr_rec2_w = None
        self.base_target2_loss = 999999
        self.brisque_metric = BRISQUE()
        self.brisque_min = 999999
        self.brisque_baseline = None

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
        # ref_rec_dr, _, _ = self.en_model(ref_rec_lr_haar_hf, ref_rec_lr_haar_hf)
        ref_rec_dr, _, _ = self.en_model(ref_rec_lr, ref_rec_lr)
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

        self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr, self.tar_hr_rec, tar_lr_rec, target_hr_base)
        if self.iter % self.conf.evaluate_iters == 0:
            self.cal_whole_image_loss(is_dn=False)
            self.show_learning_rate()

    def update_learner(self, sr=False, en=True, dn=False, matrix=False, loss_dr=None, loss_ref=None, losses=None):
        if not self.conf.dn_freeze and dn:
            self.optimizer_Dn.step()
        if en:
            self.optimizer_En.step()
        if sr:
            self.optimizer_Up.step()

        en_scheduler = self.scheduler_En
        sr_scheduler = self.scheduler_Up

        if not self.conf.dn_freeze and dn:
            self.update_dn_lrs()

        if matrix:
            en_scheduler = self.matrix_scheduler_En
            sr_scheduler = self.matrix_scheduler_Up
            if en:
                self.update_lrs(self.optimizer_En, en_scheduler, name='En', matrix=loss_dr)
            if sr:
                self.update_lrs(self.optimizer_Up, sr_scheduler, name='Up', matrix=loss_ref)
        else:
            if en:
                self.update_lrs(self.optimizer_En, en_scheduler, name='En')
            if sr:
                self.update_lrs(self.optimizer_Up, sr_scheduler, name='Up')

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        # self.save_whole_image()
        with torch.no_grad():
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
                elif loss_tar_lr < self.base_target_loss and total_loss < self.best_sr_loss and brisque_score < self.brisque_baseline:
                    self.best_sr_loss = total_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))

                if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                    self.max_psnr_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

                # self.save_whole_image(is_dn=False)

    def plot_eval(self, ref_rec_lr, ref_rec_hr, ref_dr, tar_hr_rec, tar_lr_rec, target_hr_base):
        if self.iter % self.conf.evaluate_iters == 0:
            # ref branch 
            ref_img = tensor2im(self.ref_hr)
            ref_rec_lr_img = tensor2im(ref_rec_lr)
            ref_rec_hr_img = tensor2im(ref_rec_hr)

            # tar branch
            tar_img = tensor2im(self.tar_lr)
            tar_hr_rec_img = tensor2im(tar_hr_rec)
            tar_lr_rec_img = tensor2im(tar_lr_rec)
            curr_k_np = calc_curr_k(self.dn_model.parameters()).detach().cpu().numpy()
            target_hr_base_img = tensor2im(target_hr_base)


            # save image
            fig, axs = plt.subplots(3, 3, figsize=(15, 10))
            axs[0,0].imshow(ref_img)
            axs[0,0].set_title(f'ref_img_{ref_img.shape}')
            axs[0,0].axis('off')

            axs[0,1].imshow(ref_rec_lr_img)
            axs[0,1].set_title(f'ref_rec_lr_img_{ref_rec_lr_img.shape}')
            axs[0,1].axis('off')

            axs[0,2].imshow(ref_rec_hr_img)
            axs[0,2].set_title(f'ref_rec_hr_img_{ref_rec_hr_img.shape}')
            axs[0,2].axis('off')

            axs[1,0].imshow(tar_img)
            axs[1,0].set_title(f'tar_img_{tar_img.shape}')
            axs[1,0].axis('off')

            axs[1,1].imshow(tar_hr_rec_img)
            axs[1,1].set_title(f'tar_hr_rec_img_{tar_hr_rec_img.shape}')
            axs[1,1].axis('off')

            axs[1,2].imshow(tar_lr_rec_img)
            axs[1,2].set_title(f'tar_lr_rec_img{tar_lr_rec_img.shape}')
            axs[1,2].axis('off')

            axs[2,0].imshow(target_hr_base_img)
            axs[2,0].set_title(f'target_hr_base_img_{target_hr_base_img.shape}')
            axs[2,0].axis('off')

            # Plot the second kernel
            axs[2,1].imshow(curr_k_np, cmap='gray')
            axs[2,1].set_title(f'curr_k_{curr_k_np.shape}')
            axs[2,1].axis('off')

            axs[2,2].imshow(self.kernel, cmap='gray')
            axs[2,2].set_title(f'GT_K_{self.kernel.shape}')
            axs[2,2].axis('off')

            filename = os.path.join(self.save_path, f'{self.iter}_kernel_{self.conf.scale}_UP.png' ) 
            plt.savefig(filename)
            plt.close()


            if hasattr(self.dn_model, 'parameters'):
                self.dn_model.eval()
            self.en_model.eval()
            self.sr_model.eval()
            self.dn_evaluation(is_dn=False)
            # remove saving ref rec images
            # self.sr_evaluation()

            self.eval_logger.info(f'{self.iter}, {format(self.tar_hr_psnr, ".5f")}, {format(self.tar_lr_psnr, ".5f")}, '
                                  f'{format(self.ref_hr_psnr, ".5f")}, {format(self.ref_lr_psnr, ".5f")}, '
                                  f'{format(self.tar_gt_lr_psnr, ".5f")}, {format(self.ref_gt_hr_psnr, ".5f")}')

    def dn_evaluation(self, is_dn=True):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # get dr
            self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
            # inference SR
            self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)

            torch.cuda.empty_cache()

            # DownSample with SR result
            self.tar_rec_lr_w = self.dn_model(self.tar_hr_rec_w)
            # DownSample with GT
            self.tar_gt_rec_lr_w = self.dn_model(self.tar_hr_gt_w)

            # dual cycle training
            tar_rec_lr_dr = self.en_model(self.tar_rec_lr_w, self.tar_rec_lr_w)
            self.tar_hr_rec2_w = self.sr_model(self.tar_rec_lr_w, tar_rec_lr_dr)

        self.tar_hr_psnr = calculate_psnr(self.tar_hr_rec_w, self.tar_hr_gt_w)
        self.tar_lr_psnr = calculate_psnr(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
        self.tar_gt_lr_psnr = calculate_psnr(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))

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
        target_baseline_psnr = calculate_psnr(tar_gt_w, tar_hr_rec_w)
        target_dn_psnr = 0
        if self.conf.kernel_gt_dir and self.dn_gt is not None:
            target_dn_psnr = calculate_psnr(tar_gt_dn_w, shave_a2b(tar_lr_w, tar_gt_dn_w))
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

