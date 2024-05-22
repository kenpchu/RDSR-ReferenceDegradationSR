import torch
import os
import numpy as np

from trainer.rdsrbasetrainer import RDSRBaseTrainer
from utils.img_utils import DownScale, kernel_preprocessing, DegradationProcessing
from utils.img_utils import shave_a2b, tensor2im, calculate_psnr, calc_curr_k, calculate_psnr3
from utils.img_utils import kernel_preprocessing_without_scale, kernel_preprocessing_without_pad
from PIL import Image
from networks.upsample import make_dasr_network
from networks.downsample import make_downsample_network, make_downsample_x4_network
from loss.dnloss import DownSampleRegularization
from loss.loss import CharbonnierLossV2
import matplotlib.pyplot as plt


class RDSRTrainer(RDSRBaseTrainer):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRTrainer, self).__init__(conf, tb_logger, test_dataloader,
                                          filename=filename, timestamp=timestamp, kernel=kernel)
        self.init_kernel(kernel)
        # downsample network: Degrade with GT kernel
        # self.dn_model = DownScale(self.kernel)
        self.dn_gt = DownScale(self.kernel, stride=conf.scale)

        test_data = next(iter(self.test_loader))
        self.tar_lr_w = test_data['Target_Img']
        self.tar_hr_gt_w = test_data['Target_Gt']

        self.tar_hr_rec_w = None
        self.tar_rec_lr_w = None
        self.tar_gt_rec_lr_w = None
        self.tar_lr_dr = None

        self.ref_hr_w = None
        self.ref_lr_rec_w = None
        self.ref_hr_rec_w = None

        # self.best_target_loss = 999999
        self.base_target_loss = 999999

        self.dn_model = make_downsample_network(conf).to(self.device)
        if conf.scale == 4:
            self.dn_model = make_downsample_x4_network(conf).to(self.device)

        self.baseline_model = make_dasr_network(conf).to(self.device)
        self.finetune_model = make_dasr_network(conf).to(self.device)
        self.sr_model = self.finetune_model.G
        # self.sr_model = dasr_model
        self.en_model = self.finetune_model.E
        self.sr_model.train()
        self.en_model.train()

        self.lr_l1_loss = CharbonnierLossV2().cuda()
        self.dn_regularization = DownSampleRegularization(1.0 / self.scale, self.conf.blur_kernel)
        self.curr_k = None

        self.init_optimizer()
        self.init_scheduler()

        self.insert_constraints = True
        pass

    def init_kernel(self, kernel_path):
        if kernel_path:
            self.kernel = kernel_preprocessing(kernel_path, self.conf.scale)
            # 47.9633
            # self.kernel = kernel_preprocessing_without_scale(kernel_path)
            # 32.489

    def get_target_baseline_representation(self):
        self.baseline_model.E.training = False
        with torch.no_grad():
            target_dr = self.baseline_model.E(self.tar_lr, self.tar_lr)
        return target_dr

    def save_input_img(self):
        self.save_img(self.ref_hr, 'ref_hr')
        self.save_img(self.tar_lr, 'tar_lr')

    
    def start_train_dn(self, target_hr_base):
        self.iter_step()

        # use pretrain model as GT for training
        # TODO: use eval() mode? use baseline model
        # with torch.no_grad():
        #     tar_lr_dr = self.en_model(self.tar_lr, self.tar_lr)
        #     tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)

        self.dn_model.train()
        self.optimizer_Dn.zero_grad()
        # tar_rec_lr = self.dn_model(tar_hr_rec)
        tar_rec_lr = self.dn_model(target_hr_base)

        # calculate L1 loss
        total_loss = 0
        # dn_l1_loss = self.lr_l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        dn_l1_loss = self.l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        total_loss += dn_l1_loss

        # calculate down sample regularization loss
        self.curr_k = calc_curr_k(self.dn_model.parameters())
        dn_regularization_loss = self.dn_regularization(self.curr_k, target_hr_base, tar_rec_lr)
        total_loss += dn_regularization_loss * self.conf.dn_regular_lambda

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
            tar_img = tensor2im(self.tar_lr)
            target_hr_base_img = tensor2im(target_hr_base)
            tar_rec_lr_img = tensor2im(tar_rec_lr)
            curr_k_np = self.curr_k.detach().cpu().numpy()


            fig, axs = plt.subplots(2, 3, figsize=(15, 8))

            # Plot the first kernel
            axs[0,0].imshow(curr_k_np, cmap='gray')
            axs[0,0].set_title(f'curr_K_{curr_k_np.shape}')
            axs[0,0].axis('off')

            # Plot the second kernel
            axs[0,1].imshow(self.kernel, cmap='gray')
            axs[0,1].set_title(f'GT_K_{self.kernel.shape}')
            axs[0,1].axis('off')

            axs[0,2].imshow(tar_img)
            axs[0,2].set_title(f'tar_img_{tar_img.shape}')
            axs[0,2].axis('off')
            
            axs[1,0].imshow(target_hr_base_img)
            axs[1,0].set_title(f'target_hr_base_{target_hr_base_img.shape}')
            axs[1,0].axis('off')

            axs[1,1].imshow(tar_rec_lr_img)
            axs[1,1].set_title(f'tar_rec_lr_img_{tar_rec_lr_img.shape}')
            axs[1,1].axis('off')

            filename = os.path.join(self.save_path, f'{self.iter}_kernel_{self.conf.scale}_dn.png' )
            plt.savefig(filename)
            plt.close()

            # tar_img.save(os.path.join(self.save_path, 'tar_img.png'))
            # tar_hr_img.save(os.path.join(self.save_path, 'tar_hr_img.png'))
            # tar_rec_lr_img.save(os.path.join(self.save_path, 'tar_lr_rec.png'))
            # self.dn_model_evaluation()
            self.dn_model.eval()
            self.dn_evaluation()

        if self.iter % self.conf.evaluate_iters == 0:
            # self.dn_evaluation()
            self.eval_logger.info(f'{self.iter}, {format(self.tar_hr_psnr, ".5f")}, {format(self.tar_lr_psnr, ".5f")}, '
                                  f'{format(self.tar_gt_lr_psnr, ".5f")}, {format(self.gt_kernel_psnr, ".5f")}')

            self.cal_whole_image_loss(is_dn=True)
            self.show_learning_rate(is_dn=True)

    def start_train_dn_v2(self, target_hr_base):
        self.iter_step()

        # use pretrain model as GT for training
        # TODO: use eval() mode? use baseline model
        # with torch.no_grad():
        #     tar_lr_dr = self.en_model(self.tar_lr, self.tar_lr)
        #     tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)

        self.dn_model.train()
        self.optimizer_Dn.zero_grad()
        # tar_rec_lr = self.dn_model(tar_hr_rec)
        tar_rec_lr = self.dn_model(target_hr_base)

        # calculate L1 loss
        total_loss = 0
        # dn_l1_loss = self.lr_l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        dn_l1_loss = self.l1_loss(tar_rec_lr, shave_a2b(self.tar_lr, tar_rec_lr))
        total_loss += dn_l1_loss

        # calculate down sample regularization loss
        self.curr_k = calc_curr_k(self.dn_model.parameters())
        dn_regularization_loss = self.dn_regularization(self.curr_k, target_hr_base, tar_rec_lr)
        total_loss += dn_regularization_loss * self.conf.dn_regular_lambda

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

            self.cal_whole_image_loss_dn_v2()
            self.update_dn_lambda()
            self.show_learning_rate(is_dn=True)

    def start_train_rdsr(self, target_dr, target_hr_base, sr=False, en=True, matrix=False):
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
        tar_hr_rec = self.sr_model(self.tar_lr, tar_lr_dr)
        tar_lr_rec = self.dn_model(target_hr_base)

        loss_dr = self.l1_loss(ref_rec_dr, target_dr)
        loss_ref = self.l1_loss(ref_rec_hr, shave_a2b(self.ref_hr, ref_rec_hr))

        total_loss = loss_ref * self.conf.ref_lambda
        total_loss += loss_dr * self.conf.dr_lambda
        loss_tar_sr = self.l1_loss(tar_hr_rec, target_hr_base)

        # TODO: implement dn model by using gt kernel
        loss_tar_lr = self.l1_loss(tar_lr_rec, shave_a2b(self.tar_lr, tar_lr_rec))

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
            loss_interpo = self.interpo_loss.forward(self.tar_lr, tar_hr_rec)
            total_loss += loss_interpo * self.conf.interpo_lambda

        loss_tv = 0
        if self.conf.tv_lambda != 0:
            loss_tv = self.tv_loss.forward(tar_hr_rec)
            total_loss += loss_tv * self.conf.tv_lambda

        loss_color = 0
        if self.conf.color_lambda != 0:
            loss_color = self.color_loss.forward(self.tar_lr, tar_hr_rec)
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

        self.plot_eval(ref_rec_lr, ref_rec_hr, ref_rec_dr, tar_hr_rec, tar_lr_rec)
        if self.iter % self.conf.evaluate_iters == 0:
            self.cal_whole_image_loss(is_dn=False)
            self.show_learning_rate()

    def update_learner(self, sr=False, en=True, matrix=False, loss_dr=None, loss_ref=None, losses=None):
        # TODO: dynamic changing lambda by all losses
        if not self.conf.dn_freeze:
            self.optimizer_Dn.step()
        if en:
            self.optimizer_En.step()
        if sr:
            self.optimizer_Up.step()

        en_scheduler = self.scheduler_En
        sr_scheduler = self.scheduler_Up

        if not self.conf.dn_freeze:
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

    def dn_evaluation(self, is_dn=True):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # get dr
            self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
            # inference SR
            self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)
            # DownSample with SR result
            self.tar_rec_lr_w = self.dn_model(self.tar_hr_rec_w)
            # DownSample with GT
            self.tar_gt_rec_lr_w = self.dn_model(self.tar_hr_gt_w)

        self.tar_hr_psnr = calculate_psnr(self.tar_hr_rec_w, self.tar_hr_gt_w)
        self.tar_lr_psnr = calculate_psnr(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
        self.tar_gt_lr_psnr = calculate_psnr(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))

        # GT kernel PSNR
        if self.curr_k is None:
            self.curr_k = calc_curr_k(self.dn_model.parameters())
        self.gt_kernel_psnr = calculate_psnr3(self.curr_k, self.kernel)

        if not is_dn:
            self.sr_psnr_list.append(self.tar_hr_psnr)
        else:
            self.lr_psnr_list.append(self.tar_lr_psnr)
            self.gt_lr_psnr_list.append(self.tar_gt_lr_psnr)

    def sr_evaluation(self, is_ref_loss=True):
        # test_data = next(iter(self.test_loader))
        # self.ref_hr_w = test_data['Ref_Img']

        # TODO: Need to implement all ref images to evaluate performance, current only one.
        # ref_lr_gt_w = None
        # if 'Ref_Gt' in test_data.keys():
        #     ref_lr_gt_w = test_data['Ref_Gt']

        self.ref_hr_w = []
        self.ref_lr_rec_w = []
        self.ref_hr_rec_w = []
        self.ref_hr_psnr = []
        with torch.no_grad():
            for test_data in self.test_loader:
                ref_hr_w = test_data['Ref_Img']
                ref_lr_rec_w = self.dn_model(ref_hr_w)
                ref_dr = self.en_model(ref_lr_rec_w, ref_lr_rec_w)
                ref_hr_rec_w = self.sr_model(ref_lr_rec_w, ref_dr)
                ref_hr_psnr = calculate_psnr(ref_hr_rec_w, shave_a2b(ref_hr_w, ref_hr_rec_w))

                self.ref_hr_w.append(ref_hr_w)
                self.ref_lr_rec_w.append(ref_lr_rec_w)
                self.ref_hr_rec_w.append(ref_hr_rec_w)
                self.ref_hr_psnr.append(ref_hr_psnr)

        ref_psnr_sum = 0.0
        for tmp_psnr in self.ref_hr_psnr:
            ref_psnr_sum += tmp_psnr
        ref_psnr_sum += ref_psnr_sum / len(self.ref_hr_psnr)
        self.ref_hr_psnr = ref_psnr_sum

            # almost no use
            # if ref_lr_gt_w is not None:
            #     ref_gt_hr_rec_w = self.sr_model(ref_lr_gt_w)

        # almost no use
        # if ref_lr_gt_w is not None:
        #     self.ref_lr_psnr = calculate_psnr(self.ref_lr_rec_w, shave_a2b(ref_lr_gt_w, self.ref_lr_rec_w))
        #     self.ref_gt_hr_psnr = calculate_psnr(ref_gt_hr_rec_w, shave_a2b(self.ref_hr_w, ref_gt_hr_rec_w))

    def cal_whole_image_loss_dn_v2(self):
        target_sr_loss = 0
        target_lr_loss = 0
        self.save_whole_image()
        with torch.no_grad():
            loss_tar_lr = self.l1_loss(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
            loss_tar_lr_vgg = self.vgg_loss.forward(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

            # calculate down sample regularization loss
            self.curr_k = calc_curr_k(self.dn_model.parameters())
            loss_dn_regularization = self.dn_regularization(self.curr_k, self.target_baseline_hr, self.tar_rec_lr_w)
            loss_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
            loss_dn_kernel = loss_dn_regularization - loss_dn_bq

            loss_gt_tar_lr = self.l1_loss(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))
            loss_gt_dn_regularization = self.dn_regularization(self.curr_k, self.tar_hr_gt_w, self.tar_gt_rec_lr_w)
            loss_gt_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
            loss_gt_dn_kernel = loss_gt_dn_regularization - loss_gt_dn_bq

            # total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_regularization
            # total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_kernel
            total_loss = loss_tar_lr
            self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                                  f'{format(loss_dn_bq, ".5f")}, {format(loss_dn_kernel, ".5f")}, '
                                  f'{format(loss_gt_tar_lr, ".5f")}, {format(loss_gt_dn_bq, ".5f")}, {format(loss_gt_dn_kernel, ".5f")}, '
                                  f'{format(self.dn_regularization.loss_boundaries, ".5f")}, {format(self.dn_regularization.loss_sum2one, ".5f")}, '
                                  f'{format(self.dn_regularization.loss_centralized, ".5f")}, {format(self.dn_regularization.loss_sparse, ".5f")}, '
                                  f'{format(total_loss, ".5f")}')

            tar_rec_lr_w_img = Image.fromarray(tensor2im(self.tar_rec_lr_w))
            if self.iter > self.conf.best_thres and total_loss < self.best_dn_loss:
                self.logger.info(f'Find Better Down Sample Loss at {self.iter}, total_loss: {total_loss}')
                self.best_dn_loss = total_loss
                self.min_loss_dn_psnr = self.tar_lr_psnr
                self.min_loss_dn_iter = self.iter
                self.save_model(best=True, dn_model=True)

                tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_min_loss_w.png'))

            if max(self.lr_psnr_list) == self.tar_lr_psnr:
                self.max_psnr_dn_iter = self.iter
                tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_max_psnr_w.png'))

            self.save_whole_image(is_dn=True)

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        self.save_whole_image()
        with torch.no_grad():
            loss_tar_lr = self.l1_loss(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
            loss_tar_lr_vgg = self.vgg_loss.forward(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

            # calculate down sample regularization loss
            self.curr_k = calc_curr_k(self.dn_model.parameters())
            loss_dn_regularization = self.dn_regularization(self.curr_k, self.target_baseline_hr, self.tar_rec_lr_w)
            loss_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
            loss_dn_kernel = loss_dn_regularization - loss_dn_bq

            loss_gt_tar_lr = self.l1_loss(self.tar_gt_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w))
            loss_gt_dn_regularization = self.dn_regularization(self.curr_k, self.tar_hr_gt_w, self.tar_gt_rec_lr_w)
            loss_gt_dn_bq = self.dn_regularization.loss_bicubic * self.dn_regularization.lambda_bicubic
            loss_gt_dn_kernel = loss_gt_dn_regularization - loss_gt_dn_bq

            if is_dn:
                # total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_regularization
                total_loss = loss_tar_lr + loss_tar_lr_vgg + loss_dn_kernel
                self.loss_logger.info(f'{self.iter}-whole, {format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                                      f'{format(loss_dn_bq, ".5f")}, {format(loss_dn_kernel, ".5f")}, '
                                      f'{format(loss_gt_tar_lr, ".5f")}, {format(loss_gt_dn_bq, ".5f")}, {format(loss_gt_dn_kernel, ".5f")}, '
                                      f'{format(self.dn_regularization.loss_boundaries, ".5f")}, {format(self.dn_regularization.loss_sum2one, ".5f")}, '
                                      f'{format(self.dn_regularization.loss_centralized, ".5f")}, {format(self.dn_regularization.loss_sparse, ".5f")}, '
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

                target_lr_loss += loss_tar_lr
                target_lr_loss += loss_tar_lr_vgg

                target_sr_loss += loss_tar_sr
                target_sr_loss += loss_tar_sr_vgg

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
                                      f'{format(loss_tar_lr_vgg, ".5f")}, {format(loss_tar_hf, ".5f")}, '
                                      f'{format(loss_ref, ".5f")}, {format(loss_ref_vgg, ".5f")}, '
                                      f'{format(loss_interpo, ".5f")}, {format(loss_tv, ".5f")}, '
                                      f'{format(loss_ref_hf, ".5f")}, {format(loss_ref_gv, ".5f")}, '
                                      f'{format(total_loss, ".5f")}')

                tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))

                if self.sr_iter == 0:
                    self.base_target_loss = target_lr_loss
                    self.logger.info(f'base_target_loss: {self.base_target_loss}')

                if self.iter >= self.conf.best_thres and target_lr_loss < self.base_target_loss and total_loss < self.best_sr_loss:
                    self.best_target_loss = target_lr_loss
                    self.best_sr_loss = total_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))
                    pass

                # if self.iter >= self.conf.best_thres and target_lr_loss < self.best_target_loss and total_loss < self.best_sr_loss:
                #     self.logger.info(f'Find Better Loss at {self.iter}, total_loss: {total_loss}')
                #     if self.best_target_loss != 999999 or self.best_sr_loss != 999999:
                #         # self.best_target_loss = target_lr_loss
                #         self.best_sr_loss = total_loss
                #         self.save_model(best=True, dn_model=False)
                #         self.min_loss_sr_psnr = self.tar_hr_psnr
                #         self.min_loss_sr_iter = self.iter
                #         tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))
                #     self.best_target_loss = target_lr_loss

                # TODO: notice best_thres have to match conf.evaluate_iters
                if self.sr_iter >= self.conf.best_thres and self.min_loss_sr_psnr == 0 and self.min_loss_sr_iter <= 0:
                    self.logger.info(f'Set Base Model at {self.iter}, total_loss: {total_loss}')
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_base_w.png'))

                if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                    self.max_psnr_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

                self.save_whole_image(is_dn=False)

    def save_whole_image(self, is_dn=True):
        tar_rec_lr_w_img = Image.fromarray(tensor2im(self.tar_rec_lr_w))
        tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_w.png'))
        if is_dn:
            tar_gt_rec_lr_w_img = Image.fromarray(tensor2im(self.tar_gt_rec_lr_w))
            tar_gt_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_gt_rec_lr_w.png'))
        else:
            tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))
            tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_w.png'))

            for i in range(len(self.ref_hr_rec_w)):
                ref_hr_rec_w_img = Image.fromarray(tensor2im(self.ref_hr_rec_w[i]))
                ref_hr_rec_w_img.save(os.path.join(self.save_path, f'ref_hr_rec_{i+1}_w.png'))
                ref_lr_rec_w_img = Image.fromarray(tensor2im(self.ref_lr_rec_w[i]))
                ref_lr_rec_w_img.save(os.path.join(self.save_path, f'ref_lr_rec_{i+1}_w.png'))

    def update_dn_lambda(self):
        # TODO: waiting for refactor
        bic_loss_to_start_change = 0.4
        lambda_update_freq = 200
        lambda_bicubic_min = 5e-6
        lambda_bicubic_decay_rate = 100.
        lambda_sparse_end = 5
        lambda_centralized_end = 1
        # lambda_sparse_end = 3
        # lambda_centralized_end = 3
        kernel_regularization = 2000

        if self.insert_constraints and self.iter > kernel_regularization:
            self.conf.dn_regular_lambda = 10
            self.dn_regularization.loss_bicubic = 0
            self.dn_regularization.lambda_centralized = lambda_centralized_end
            self.dn_regularization.lambda_sparse = lambda_sparse_end
            self.insert_constraints = False


'''
    # this is for debug SR issue 
    def set_baseline_img(self):
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']
        tar_gt_w = test_data['Target_Gt']
        self.baseline_model.eval()
        self.en_model.eval()
        self.sr_model.eval()
        with torch.no_grad():
            tar_hr_dr_w = self.baseline_model.E(tar_lr_w, tar_lr_w)
            tar_hr_rec_w_1 = self.baseline_model.G(tar_lr_w, tar_hr_dr_w)
            tar_hr_dr_w = self.en_model(tar_lr_w, tar_lr_w)
            tar_hr_rec_w_2 = self.sr_model(tar_lr_w, tar_hr_dr_w)
            self.en_model.train()
            self.sr_model.train()
            tar_hr_dr_w, _, _ = self.en_model(tar_lr_w, tar_lr_w)
            tar_hr_rec_w_3 = self.sr_model(tar_lr_w, tar_hr_dr_w)

        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w_1))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'target_hr_baseline_train_w_1.png'))

        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w_2))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'target_hr_baseline_train_w_2.png'))

        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w_3))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'target_hr_baseline_train_w_3.png'))

        self.baseline_model.eval()
        with torch.no_grad():
            tar_hr_rec_w = self.baseline_model(tar_lr_w)
        self.target_baseline_hr = tar_hr_rec_w

        # calculate baseline image PSNR
        target_baseline_psnr = calculate_psnr(tar_gt_w, tar_hr_rec_w)
        self.baseline_psnr = target_baseline_psnr
        target_baseline_psnr = format(target_baseline_psnr, '.5f')
        self.logger.info(f"Target Baseline PSNR:{target_baseline_psnr}")
        self.eval_logger.info(f"{self.iter}, {target_baseline_psnr}")

        # save baseline image
        tar_hr_rec_w_img = Image.fromarray(tensor2im(tar_hr_rec_w))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'target_hr_baseline_w.png'))

        tar_lr_w_path = os.path.join(self.save_path, 'target_lr_w.png')
        tar_lr_w_img = Image.fromarray(tensor2im(tar_lr_w))
        tar_lr_w_img.save(tar_lr_w_path)

        tar_hr_gt_w_path = os.path.join(self.save_path, 'target_hr_gt_w.png')
        tar_hr_gt_w_img = Image.fromarray(tensor2im(tar_gt_w))
        tar_hr_gt_w_img.save(tar_hr_gt_w_path)
'''