import torch
import os

from trainer.rdsrdisctrainerv3 import RDSRDiscTrainerV3
from utils.img_utils import DownScale, kernel_preprocessing, DegradationProcessing
from utils.img_utils import shave_a2b, tensor2im, calculate_psnr, calc_curr_k, calculate_psnr3
from utils.utils import set_requires_grad
from PIL import Image
from networks.upsample import make_dasr_network
from networks.downsample import make_down_double_network, make_downsample_x4_network
from loss.dnloss import DownSampleRegularization
from loss.loss import CharbonnierLossV2, GANLoss
from networks.upsample import make_up_discriminator_net


class RDSRDiscTrainerV4(RDSRDiscTrainerV3):
    target_decay_low_thres = 0.5
    target_decay_weight = 0.5
    save_losses_cnt = 2

    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV4, self).__init__(conf, tb_logger, test_dataloader,
                                                filename=filename, timestamp=timestamp, kernel=kernel)
        self.tar_hr_rec2_w = None
        self.base_target2_loss = 999999
        self.save_losses = []
        if conf.scale == 4:
            self.dn_model = make_down_double_network(conf).to(self.device)
            self.optimizer_Dn = torch.optim.Adam(self.dn_model.parameters(), lr=self.conf.lr_dn, betas=(self.conf.beta1, 0.999))
            self.scheduler_Dn = torch.optim.lr_scheduler.StepLR(self.optimizer_Dn, step_size=self.conf.lrs_step_size, gamma=self.conf.lrs_gamma)
            self.matrix_scheduler_Dn = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Dn, 'min',
                                                                                  factor=self.conf.lrs_gamma,
                                                                                  patience=self.conf.lrs_patience,
                                                                                  cooldown=self.conf.lrs_cooldown,
                                                                                  min_lr=self.conf.lrs_minlr, verbose=0)
        self.tar_hr_rec = None
        self.loss_disc = 999999
        self.up_disc_model = make_up_discriminator_net(self.device)
        self.optimizer_up_disc = torch.optim.Adam(self.up_disc_model.parameters(), lr=self.conf.lr_up_disc,
                                                  betas=(self.conf.beta1, 0.999))
        self.up_disc_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_up_disc, step_size=250, gamma=0.5)
        self.gan_loss = GANLoss().to(self.device)

    def start_train_up(self, target_dr, target_hr_base, sr=False, en=True, matrix=False):
        self.train_upsample(target_dr, target_hr_base, sr=sr, en=en, matrix=matrix)
        self.train_up_discriminator()

    def train_up_discriminator(self):
        set_requires_grad(self.up_disc_model, True)
        self.optimizer_up_disc.zero_grad()

        pred_fake_hr = self.up_disc_model(self.tar_hr_rec.detach())
        loss_disc_fake = self.gan_loss(pred_fake_hr, False)

        pred_hr = self.up_disc_model(self.ref_hr)
        loss_disc_true = self.gan_loss(pred_hr, True)

        self.loss_disc = (loss_disc_fake + loss_disc_true) * 0.5
        self.loss_disc.backward()
        self.optimizer_up_disc.step()
        self.up_disc_lr_scheduler.step()

    def start_train_dn(self, target_hr_base):
        self.iter_step()

        self.dn_model.train()
        self.optimizer_Dn.zero_grad()
        tar_rec_lr = self.dn_model(target_hr_base)

        # calculate L1 loss
        total_loss = 0
        dn_l1_loss = self.l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        total_loss += dn_l1_loss

        # calculate down sample regularization loss
        # self.curr_k = calc_curr_k(self.dn_model.parameters())
        # dn_regularization_loss = self.dn_regularization(self.curr_k, target_hr_base, tar_rec_lr)
        # total_loss += dn_regularization_loss * self.conf.dn_regular_lambda

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
        # self.update_dn_lrs()
        self.update_matrix_dn_lrs(total_loss)

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

        # TODO:compare results between train() and eval()
        ref_rec_lr = self.dn_model(self.ref_hr)
        ref_rec_dr, _, _ = self.en_model(ref_rec_lr, ref_rec_lr)
        ref_rec_hr = self.sr_model(ref_rec_lr, ref_rec_dr)

        tar_lr_dr, _, _ = self.en_model(self.tar_lr, self.tar_lr)
        self.tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)
        tar_lr_rec = self.dn_model(target_hr_base)

        loss_dr = self.l1_loss(ref_rec_dr, target_dr)
        loss_ref = self.l1_loss(ref_rec_hr, shave_a2b(self.ref_hr, ref_rec_hr))
        loss_tar_sr = self.l1_loss(self.tar_hr_rec, target_hr_base)
        loss_tar_lr = self.l1_loss(tar_lr_rec, shave_a2b(self.tar_lr, tar_lr_rec))

        total_loss = loss_ref * self.conf.ref_lambda
        total_loss += loss_dr * self.conf.dr_lambda
        total_loss += loss_tar_lr * self.conf.vgg_tar_lambda

        # The purpose is for bounding baseline result
        if self.conf.target_lambda != 0:
            total_loss += loss_tar_sr * self.conf.target_lambda

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
        self.update_lambda()

        self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr, self.tar_hr_rec, tar_lr_rec)
        if self.iter % self.conf.evaluate_iters == 0:
            self.cal_whole_image_loss(is_dn=False)
            self.show_learning_rate()

    def update_lambda(self):
        if (self.conf.target_thres != 0 and self.sr_iter >= self.conf.target_thres
                and self.sr_iter % self.conf.target_thres == 0):
            self.conf.target_lambda *= self.target_decay_weight
            # print(self.sr_iter)
            # print(self.conf.target_lambda)
            if self.conf.target_lambda < self.target_decay_low_thres:
                self.conf.target_lambda = 0

    def dn_evaluation(self, is_dn=True):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # get dr
            self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
            # inference SR
            self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)
            # DownSample with SR result
            torch.cuda.empty_cache()
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

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        self.save_whole_image()
        with torch.no_grad():
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

                self.save_whole_image(is_dn=True)
            else:
                loss_tar_sr = self.l1_loss(self.target_baseline_hr, self.tar_hr_rec_w)
                loss_tar_sr_vgg = self.vgg_loss.forward(self.tar_hr_rec_w, shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w))

                loss_interpo = self.interpo_loss.forward(self.tar_lr_w, self.tar_hr_rec_w)
                loss_tv = self.tv_loss.forward(self.tar_hr_rec_w)
                loss_tar_hf = self.hf_loss.forward(shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w), self.tar_hr_rec_w)

                # add for set the baseline
                loss_tar_sr2 = self.l1_loss(shave_a2b(self.target_baseline_hr, self.tar_hr_rec2_w), self.tar_hr_rec2_w)

                target_lr_loss += loss_tar_lr
                target_lr_loss += loss_tar_lr_vgg * self.conf.vgg_lambda

                target_sr_loss += loss_tar_sr
                target_sr_loss += loss_tar_sr_vgg * self.conf.vgg_lambda

                # try lr: 3, sr: 1
                target_total_loss = target_lr_loss * self.conf.target_lr_lambda + target_sr_loss * self.conf.target_sr_lambda

                loss_ref = 0
                loss_ref_vgg = 0
                loss_ref_hf = 0
                loss_ref_gv = 0
                for i in range(len(self.ref_hr_w)):
                    loss_ref += self.l1_loss(shave_a2b(self.ref_hr_w[i], self.ref_hr_rec_w[i]), self.ref_hr_rec_w[i])
                    loss_ref_vgg += self.vgg_loss.forward(shave_a2b(self.ref_hr_w[i], self.ref_hr_rec_w[i]), self.ref_hr_rec_w[i])
                    loss_ref_hf += self.hf_loss.forward(shave_a2b(self.ref_hr_w[i], self.ref_hr_rec_w[i]), self.ref_hr_rec_w[i])
                    loss_ref_gv += self.GV_loss.forward(shave_a2b(self.ref_hr_w[i], self.ref_hr_rec_w[i]), self.ref_hr_rec_w[i])
                loss_ref /= len(self.ref_hr_w)
                loss_ref_vgg /= len(self.ref_hr_w)
                loss_ref_hf /= len(self.ref_hr_w)
                loss_ref_gv /= len(self.ref_hr_w)

                ref_total_loss = loss_ref + loss_ref_vgg

                total_loss = self.conf.total_target_lambda * target_total_loss + self.conf.total_ref_lambda * ref_total_loss

                self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_sr, ".5f")}, '
                                      f'{format(loss_tar_sr_vgg, ".5f")}, {format(loss_tar_lr, ".5f")}, '
                                      f'{format(loss_tar_lr_vgg, ".5f")}, {format(self.loss_disc, ".5f")}, '
                                      f'{format(loss_tar_sr2, ".5f")}, {format(loss_tar_hf, ".5f")}, '
                                      f'{format(loss_ref, ".5f")}, {format(loss_ref_vgg, ".5f")}, '
                                      f'{format(loss_interpo, ".5f")}, {format(loss_tv, ".5f")}, '
                                      f'{format(loss_ref_hf, ".5f")}, {format(loss_ref_gv, ".5f")}, '
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
                elif loss_tar_lr < self.base_target_loss and total_loss < self.best_sr_loss:
                    self.best_sr_loss = total_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))

                '''
                # one more sr2 condition
                if self.iter >= self.conf.best_thres and target_lr_loss < self.base_target_loss and loss_tar_sr2 < self.base_target2_loss:
                    self.save_losses.append([total_loss, self.iter, self.tar_hr_psnr])
                    self.save_losses = sorted(self.save_losses, key=lambda x: x[0])
                    if len(self.save_losses) > self.save_losses_cnt:
                        self.save_losses = self.save_losses[:self.save_losses_cnt]

                    if total_loss < self.best_sr_loss:
                        self.logger.info(f'Set Best Model at {self.iter}, total_loss: {total_loss}')
                        self.best_target_loss = target_lr_loss
                        self.best_sr_loss = total_loss
                        self.save_model(best=True, dn_model=False)
                        self.min_loss_sr_psnr = self.tar_hr_psnr
                        self.min_loss_sr_iter = self.iter
                        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))
                '''

                # TODO: notice best_thres have to match conf.evaluate_iters
                # if self.sr_iter >= self.conf.best_thres and self.min_loss_sr_psnr == 0 and self.min_loss_sr_iter <= 0:
                #     self.logger.info(f'Set Base Model at {self.iter}, total_loss: {total_loss}')
                #     self.save_model(best=True, dn_model=False)
                #     self.min_loss_sr_psnr = self.tar_hr_psnr
                #     self.min_loss_sr_iter = self.iter
                #     tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_base_w.png'))

                if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                    self.max_psnr_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

                self.save_whole_image(is_dn=False)

    def show_learning_rate(self, is_dn=False):
        curr_dn_lr = 0
        if hasattr(self.dn_model, 'parameters'):
            curr_dn_lr = self.optimizer_Dn.param_groups[0]['lr']
        if not is_dn:
            curr_en_lr = self.optimizer_En.param_groups[0]['lr']
            curr_up_lr = self.optimizer_Up.param_groups[0]['lr']
            curr_up_disc_lr = self.optimizer_up_disc.param_groups[0]['lr']
            self.lr_logger.info(f'{self.iter},{curr_dn_lr},{curr_en_lr},{curr_up_lr},{curr_up_disc_lr}')
        else:
            self.lr_logger.info(f'{self.iter},{curr_dn_lr}')

    def save_model(self, best=False, dn_model=True, name=''):
        if self.timestamp:
            output_path = os.path.join(self.conf.train_log, self.timestamp, self.filename)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.save_path = output_path
        if not best:
            dn_model_path = os.path.join(self.save_path, f'model_dn_{self.filename}_{self.iter}.pt')
            sr_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_{self.iter}.pt')
        else:
            dn_model_path = os.path.join(self.save_path, f'model_dn_{self.filename}_best.pt')
            sr_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_best.pt')
            if name:
                sr_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_{name}.pt')
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

    def finish(self):
        # save_model
        if self.conf.save_model:
            self.save_model()
        if self.conf.save_image:
            pass
        if len(self.save_losses) < self.save_losses_cnt:
            sr_best_model_path = os.path.join(self.save_path, f'model_sr_{self.filename}_best.pt')
            sr_best_img_path = os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png')
            if os.path.exists(sr_best_model_path):
                os.remove(sr_best_model_path)
            if os.path.exists(sr_best_img_path):
                os.remove(sr_best_img_path)
            self.logger.info(f'Not reach save losses count: {len(self.save_losses)}')

        self.eval_logger.info(f'min_loss_sr_psnr, max_sr_psnr,  dasr_psnr, min_loss_diff_dasr, '
                              f'max_psnr_diff_dasr, min_loss_sr_iter, max_psnr_sr_iter, '
                              f'min_loss_lr_iter, max_psnr_lr_iter')
        min_loss_diff_dasr = format(self.min_loss_sr_psnr - self.baseline_psnr, ".5f")
        max_psnr_diff_dasr = 0
        if len(self.sr_psnr_list) > 0:
            max_psnr_diff_dasr = format(max(self.sr_psnr_list) - self.baseline_psnr, ".5f")
        min_loss_sr_psnr = format(self.min_loss_sr_psnr, ".5f")
        max_sr_psnr = 0
        if len(self.sr_psnr_list) > 0:
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