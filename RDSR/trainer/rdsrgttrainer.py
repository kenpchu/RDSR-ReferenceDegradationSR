import torch
import os

from trainer.rdsrbasetrainer import RDSRBaseTrainer
from utils.img_utils import DownScale, kernel_preprocessing, DegradationProcessing, kernel_preprocessing_without_shift
from utils.img_utils import shave_a2b, tensor2im, calculate_psnr
from PIL import Image
from networks.upsample import make_dasr_network, make_downsample_network


class RDSRGtTrainer(RDSRBaseTrainer):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRGtTrainer, self).__init__(conf, tb_logger, test_dataloader,
                                            filename=filename, timestamp=timestamp, kernel=kernel)
        self.kernel_path = kernel
        self.init_kernel(kernel)
        # downsample network: Degrade with GT kernel
        self.dn_model = DownScale(self.kernel, stride=conf.scale)

        # self.dn_model_cmp = DegradationProcessing(conf, self.kernel)
        # DownScale vs DegradationProcessing -  lr: 42 vs 37, gt: 60 vs 37

        self.baseline_model = make_dasr_network(conf).to(self.device)
        self.finetune_model = make_dasr_network(conf).to(self.device)
        self.sr_model = self.finetune_model.G
        # self.sr_model = dasr_model
        self.en_model = self.finetune_model.E
        self.sr_model.train()
        self.en_model.train()

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

        self.init_optimizer()
        self.init_scheduler()
        pass

    def init_kernel(self, kernel_path):
        # kernel_degrade_inst = KernelDegrade()
        self.kernel = kernel_preprocessing(kernel_path, self.conf.scale)

    def get_target_baseline_representation(self):
        self.baseline_model.E.training = False
        with torch.no_grad():
            target_dr = self.baseline_model.E(self.tar_lr, self.tar_lr)
        return target_dr

    def start_train_rdsr(self, target_dr, target_hr_base, sr=False, en=True, matrix=False):
        self.iter_step()
        self.en_model.train()
        self.sr_model.train()
        self.sr_iter += 1
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
        if en:
            self.optimizer_En.step()
        if sr:
            self.optimizer_Up.step()

        en_scheduler = self.scheduler_En
        sr_scheduler = self.scheduler_Up

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

    def target_whole_loss(self, tar_lr_w, tar_hr_rec_w, tar_rec_lr_w, is_dn=True):
        # This function is observed for overall target image quality
        is_best_img = False

        with torch.no_grad():
            target_sr_loss = 0
            target_lr_loss = 0
            loss_tar_sr = self.l1_loss(self.target_baseline_hr, tar_hr_rec_w)
            loss_tar_lr = self.l1_loss(tar_rec_lr_w, shave_a2b(tar_lr_w, tar_rec_lr_w))

            # 0819 only use reconstruct loss to evaluate best image
            loss_tar_lr_vgg = 0
            if self.conf.vgg_tar_lambda != 0:
                loss_tar_lr_vgg = self.vgg_loss.forward(tar_rec_lr_w, shave_a2b(tar_lr_w, tar_rec_lr_w))

                # TODO: target lr unsample & target hr VGG

            loss_tar_sr_vgg = self.vgg_loss.forward(tar_hr_rec_w, shave_a2b(self.target_baseline_hr, tar_hr_rec_w))

            target_lr_loss += loss_tar_lr
            # target_lr_loss += loss_tar_lr_vgg * self.conf.vgg_tar_lambda

            target_sr_loss += loss_tar_sr
            target_sr_loss += loss_tar_sr_vgg * self.conf.vgg_tar_lambda

            total_loss = target_lr_loss * self.conf.target_lr_lambda + target_sr_loss * self.conf.target_sr_lambda

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
                                  f'{format(loss_tar_lr_vgg, ".5f")}, {format(loss_tar_lr, ".5f")}, '
                                  f'{format(total_loss, ".5f")}')

            if self.iter > self.conf.best_thres and total_loss < self.best_sr_loss:
                self.best_sr_loss = total_loss
                self.save_model(best=True, dn_model=False)
                self.save_best_img()
                self.logger.info(f'Find Better Loss at {self.iter}, total_loss: {total_loss}')
                is_best_img = True

        return is_best_img

    def dn_evaluation(self, is_dn=True):
        with torch.no_grad():
            # get dr
            self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
            # inference SR
            self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)
            # DownSample with SR result
            self.tar_rec_lr_w = self.dn_model(self.tar_hr_rec_w)

        self.tar_hr_psnr = calculate_psnr(self.tar_hr_rec_w, self.tar_hr_gt_w)
        self.tar_lr_psnr = calculate_psnr(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

        self.sr_psnr_list.append(self.tar_hr_psnr)
        self.lr_psnr_list.append(self.tar_lr_psnr)

    def sr_evaluation(self, is_ref_loss=True):
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

    def cal_whole_image_loss(self, is_dn=True):
        # This function is observed for overall target image quality
        target_sr_loss = 0
        target_lr_loss = 0
        self.save_whole_image()
        with torch.no_grad():
            loss_tar_lr = self.l1_loss(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))
            loss_tar_lr_vgg = self.vgg_loss.forward(self.tar_rec_lr_w, shave_a2b(self.tar_lr_w, self.tar_rec_lr_w))

            loss_tar_sr = self.l1_loss(self.target_baseline_hr, self.tar_hr_rec_w)
            loss_tar_sr_vgg = self.vgg_loss.forward(self.tar_hr_rec_w, shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w))

            loss_interpo = self.interpo_loss.forward(self.tar_lr_w, self.tar_hr_rec_w)
            loss_tv = self.tv_loss.forward(self.tar_hr_rec_w)
            loss_tar_hf = self.hf_loss.forward(shave_a2b(self.target_baseline_hr, self.tar_hr_rec_w), self.tar_hr_rec_w)

            target_lr_loss += loss_tar_lr
            # target_lr_loss += loss_tar_lr_vgg

            target_sr_loss += loss_tar_sr
            # target_sr_loss += loss_tar_sr_vgg

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
                                  f'{format(loss_tar_sr_vgg, ".5f")}, {format(loss_tar_hf, ".5f")}, '
                                  f'{format(loss_ref, ".5f")}, {format(loss_ref_vgg, ".5f")}, '
                                  f'{format(loss_tar_lr, ".5f")}, {format(loss_tar_lr_vgg, ".5f")}, '
                                  f'{format(loss_interpo, ".5f")}, {format(loss_tv, ".5f")}, '
                                  f'{format(loss_ref_hf, ".5f")}, {format(loss_ref_gv, ".5f")}, '
                                  f'{format(total_loss, ".5f")}')

            tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))
            if self.iter > self.conf.best_thres and target_lr_loss < self.best_target_loss and total_loss < self.best_sr_loss:
                self.logger.info(f'Find Better Loss at {self.iter}, total_loss: {total_loss}')
                if self.best_target_loss != 999999 or self.best_target_loss != 999999:
                    # self.best_target_loss = target_lr_loss
                    self.best_sr_loss = total_loss
                    self.save_model(best=True, dn_model=False)
                    self.min_loss_sr_psnr = self.tar_hr_psnr
                    self.min_loss_sr_iter = self.iter
                    tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_min_loss_w.png'))
                self.best_target_loss = target_lr_loss

            if len(self.sr_psnr_list) > 0 and max(self.sr_psnr_list) == self.tar_hr_psnr:
                self.max_psnr_sr_iter = self.iter
                tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_max_psnr_w.png'))

            self.save_whole_image(is_dn=False)

    def save_whole_image(self, is_dn=True):
        tar_rec_lr_w_img = Image.fromarray(tensor2im(self.tar_rec_lr_w))
        tar_rec_lr_w_img.save(os.path.join(self.save_path, 'tar_lr_rec_w.png'))

        tar_hr_rec_w_img = Image.fromarray(tensor2im(self.tar_hr_rec_w))
        tar_hr_rec_w_img.save(os.path.join(self.save_path, 'tar_hr_rec_w.png'))

        for i in range(len(self.ref_hr_rec_w)):
            ref_hr_rec_w_img = Image.fromarray(tensor2im(self.ref_hr_rec_w[i]))
            ref_hr_rec_w_img.save(os.path.join(self.save_path, f'ref_hr_rec_{i+1}_w.png'))
            ref_lr_rec_w_img = Image.fromarray(tensor2im(self.ref_lr_rec_w[i]))
            ref_lr_rec_w_img.save(os.path.join(self.save_path, f'ref_lr_rec_{i+1}_w.png'))

    # [DOE] GT Degradation compare
    def cmp_dn_kernel(self):
        test_data = next(iter(self.test_loader))
        tar_lr_w = test_data['Target_Img']
        tar_gt_w = test_data['Target_Gt']
        ker1 = kernel_preprocessing_without_shift(self.kernel_path, self.conf.scale)
        ker2 = kernel_preprocessing(self.kernel_path, self.conf.scale)

        # self.dn_model = DownScale(self.kernel)
        # self.dn_model_cmp = DegradationProcessing(self.conf, self.kernel)
        Dn1_ker1 = DownScale(ker1)
        Dn1_ker2 = DownScale(ker2)
        Dn2_ker1 = DegradationProcessing(self.conf, ker1)
        Dn2_ker2 = DegradationProcessing(self.conf, ker2)

        tar_rec_lr_w_1 = Dn1_ker1(tar_gt_w)
        tar_rec_lr_w_2 = Dn1_ker2(tar_gt_w)
        tar_rec_lr_w_3 = Dn2_ker1(tar_gt_w)
        tar_rec_lr_w_4 = Dn2_ker2(tar_gt_w)

        dn_psnr1 = calculate_psnr(tar_rec_lr_w_1, shave_a2b(tar_lr_w, tar_rec_lr_w_1))
        dn_psnr2 = calculate_psnr(tar_rec_lr_w_2, shave_a2b(tar_lr_w, tar_rec_lr_w_1))
        dn_psnr3 = calculate_psnr(shave_a2b(tar_rec_lr_w_3, tar_rec_lr_w_1), shave_a2b(tar_lr_w, tar_rec_lr_w_1))
        dn_psnr4 = calculate_psnr(shave_a2b(tar_rec_lr_w_4, tar_rec_lr_w_1), shave_a2b(tar_lr_w, tar_rec_lr_w_1))

        print(dn_psnr1)
        print(dn_psnr2)
        print(dn_psnr3)
        print(dn_psnr4)
        '''
        38.244631426250265
        60.606622803184955
        47.432723984008014
        38.151365059993296
        '''

# iteration: 2
# SR Total Loss: 2, total_loss: 8.678102493286133
# Find Better Loss at 2, total_loss: 1.038428544998169
# iteration: 4
# SR Total Loss: 4, total_loss: 7.380895614624023
# Find Better Loss at 4, total_loss: 0.8981901407241821
# iteration: 6
# SR Total Loss: 6, total_loss: 20.880765914916992
# Find Better Loss at 6, total_loss: 0.8478414416313171
# iteration: 8
# SR Total Loss: 8, total_loss: 14.992965698242188
