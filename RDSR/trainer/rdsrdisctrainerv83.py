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
import scipy.io

# remove haar wavlet
class RDSRDiscTrainerV83(RDSRDiscTrainerV82):
    def __init__(self, conf, tb_logger, test_dataloader, filename='', timestamp='', kernel=None):
        torch.manual_seed(conf.random_seed)
        super(RDSRDiscTrainerV83, self).__init__(conf, tb_logger, test_dataloader,
                                                 filename=filename, timestamp=timestamp, kernel=kernel)

        if kernel: 
            self.kernel = scipy.io.loadmat(kernel)['Kernel']

        self.use_gt_k = True if conf.use_gt_k !=0 else False
    


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

    def dn_evaluation(self, is_dn=True):
        torch.cuda.empty_cache()
        with torch.no_grad():
            # get dr
            self.tar_lr_dr = self.en_model(self.tar_lr_w, self.tar_lr_w)
            # inference SR
            self.tar_hr_rec_w = self.sr_model(self.tar_lr_w, self.tar_lr_dr)

            torch.cuda.empty_cache()

            # DownSample with SR result
            self.tar_rec_lr_w = self.dn_model(self.tar_hr_rec_w) if not self.use_gt_k else self.dn_gt(self.tar_hr_rec_w)
            # DownSample with GT
            self.tar_gt_rec_lr_w = self.dn_model(self.tar_hr_gt_w) if not self.use_gt_k else self.dn_gt(self.tar_hr_gt_w)

            # dual cycle training
            tar_rec_lr_dr = self.en_model(self.tar_rec_lr_w, self.tar_rec_lr_w)
            self.tar_hr_rec2_w = self.sr_model(self.tar_rec_lr_w, tar_rec_lr_dr)

        self.tar_hr_psnr = cal_y_psnr(tensor2im(self.tar_hr_rec_w), tensor2im(self.tar_hr_gt_w), self.conf.scale)
        self.tar_lr_psnr = cal_y_psnr(tensor2im(self.tar_rec_lr_w), tensor2im(shave_a2b(self.tar_lr_w, self.tar_rec_lr_w)), self.conf.scale)
        self.tar_gt_lr_psnr = cal_y_psnr(tensor2im(self.tar_gt_rec_lr_w), tensor2im(shave_a2b(self.tar_lr_w, self.tar_gt_rec_lr_w)), self.conf.scale)

        # GT kernel PSNR
        if self.conf.kernel_gt_dir:
            if self.curr_k is None:
                self.curr_k = calc_curr_k(self.dn_model.parameters()) if not self.use_gt_k else self.kernel
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