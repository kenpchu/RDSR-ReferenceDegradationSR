from options import Options, JsonOptions
from utils.utils import create_TBlogger, create_train_logger, dump_training_settings, set_loader_seed, set_seed
from datetime import datetime
from utils.img_utils import sample_ref_by_color_space, kernel_processing, save_ref_img

from data.data import gen_test_dataloader, gen_train_dataloader_adaptive, gen_dn_train_dataloader
# from trainer import RDSRTrainer
from trainer.rdsrdisctrainerv22 import RDSRDiscTrainerV2

import os
import glob
import random
import time
import torch
import numpy as np

# torch.backends.cudnn.benchmark=False
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def ref_preprocessing(conf, ref_path_list, tar_ind, tar_path_list, gt_ref=''):
    random.seed(conf.random_seed)
    if not conf.ref_pool:
        ref_path = ref_path_list[tar_ind]
        if os.path.isdir(ref_path):
            ref_list = sorted(os.listdir(ref_path))
            ref_list = [os.path.join(ref_path, p) for p in ref_list]

            ref_idx_list = random.sample(range(len(ref_list)), conf.ref_count)
            ref_list = [ref_list[idx] for idx in ref_idx_list]
        else:
            ref_list = [ref_path]
    else:
        start_ind = conf.ref_st_ind
        if conf.ref_st_ind >= len(ref_path_list) - 1:
            start_ind = len(ref_path_list) - 1
        ref_sample_path_list = ref_path_list[start_ind:start_ind + conf.ref_limit]
        # TODO: add function to select reference sample images
        if conf.ref_selection:
            ref_idx_list = sample_ref_by_color_space(tar_path_list[tar_ind], ref_sample_path_list, ref_cnt=conf.ref_count)
        else:
            ref_idx_list = random.sample(range(len(ref_sample_path_list)), conf.ref_count)
        ref_list = [ref_sample_path_list[idx] for idx in ref_idx_list]

    print(ref_list)

    # set ref gt images if exist
    ref_gt_list = ''
    if gt_ref:
        ref_gt_path = gt_ref[tar_ind]
        if os.path.isdir(ref_gt_path):
            ref_gt_list = sorted(os.listdir(ref_gt_path))
        ref_gt_list = [os.path.join(ref_gt_path, p) for p in ref_gt_list]
        ref_gt_list = [ref_gt_list[idx] for idx in ref_idx_list]

    return ref_list, ref_gt_list


def selection_mode(conf, tar_path_list, ref_path_list, gt_tar, gt_ker):
    if conf.target_select_path and os.path.exists(conf.target_select_path):
        with open(conf.target_select_path, 'r') as fp:
            tar_list_str = fp.read()
            # follow by format
            tar_tmp_list = []
            tar_gt_tmp_list = []
            ker_tmp_list = []

            tar_list = tar_list_str.split(',')
            for tar_name in tar_list:
                for i, tar_path in enumerate(tar_path_list):
                    if os.path.basename(tar_path) == tar_name.strip():
                        tar_tmp_list.append(tar_path)
                        tar_gt_tmp_list.append(gt_tar[i])
                        ker_tmp_list.append(gt_ker[i])

            tar_path_list = tar_tmp_list
            gt_tar = tar_gt_tmp_list
            gt_ker = ker_tmp_list

    if conf.ref_select_path:
        ref_dir_path = os.path.dirname(ref_path_list[0])
        ref_sel_path_list = sorted(glob.glob(os.path.join(conf.ref_select_path, "*_ref.txt")))
        ref_sel_path_list = sorted(ref_sel_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1]))
        ref_all_list = []
        for tar_path in tar_path_list:
            tar, ext = os.path.splitext(os.path.basename(tar_path))
            ref_tmp_list = []
            for ref_tmp_path in ref_sel_path_list:
                ref_str, ext = os.path.splitext(os.path.basename(ref_tmp_path))
                ref = '_'.join(ref_str.split('_')[:2])
                if tar == ref:
                    with open(ref_tmp_path, 'r') as fp:
                        ref_str_list = fp.read().split(',')
                        ref_str_list = [ref.strip() for ref in ref_str_list]
                        for tmp_ref in ref_str_list:
                            ref_tmp_list.append(os.path.join(ref_dir_path, tmp_ref))
            assert len(ref_tmp_list) > 0
            ref_all_list.append(ref_tmp_list)
            ref_path_list = ref_all_list

    return tar_path_list, ref_path_list, gt_tar, gt_ker


def train(conf, tb_logger, tar_path_list, ref_path_list, timestamp='', gt_tar='', gt_ker='', gt_ref=''):
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    if len(gt_ref) == 0:
        gt_ref = ''

    # this function is for target & ref select path
    tar_path_list, ref_path_list, gt_tar, gt_ker = selection_mode(conf, tar_path_list, ref_path_list, gt_tar, gt_ker)

    # Add LR pretrain model
    if conf.pretrained_lr_path:
        lr_model_path_list = glob.glob(os.path.join(conf.pretrained_lr_path, "*.pt"))
        lr_model_path_list = sorted(lr_model_path_list, key=lambda x: int(x.split("_")[-2]))
        lr_model_path_list = lr_model_path_list[conf.target_ind:conf.target_ind + conf.target_count]
        print(lr_model_path_list)

    origin_iter = conf.train_iters
    origin_ref_cnt = conf.ref_count
    for idx, target_path in enumerate(tar_path_list):
        # set gt target & gt kernel
        if gt_tar:
            target_gt_path = gt_tar[idx]

        ker_gt_path = ''
        if gt_ker:
            ker_gt_path = gt_ker[idx]

        # set reference images
        conf.ref_count = origin_ref_cnt

        if conf.target_select_path or conf.ref_select_path:
            ref_list = ref_path_list[idx]
            ref_gt_list = gt_ref
        else:
            ref_list, ref_gt_list = ref_preprocessing(conf, ref_path_list, idx, tar_path_list, gt_ref=gt_ref)

        # setup test dataset
        test_dataloader = gen_test_dataloader(
            conf, target_path, ref_list, tar_gt=target_gt_path, ref_gt=ref_gt_list)

        # for _ in range(10):
        #     print(next(iter(test_dataloader)))

        tar, ext = os.path.splitext(os.path.basename(target_path))
        tar_name = '_'.join(tar.split('_')[:2])

        # TODO: to check trainer process
        trainer = RDSRDiscTrainerV2(conf, tb_logger, test_dataloader, kernel=ker_gt_path, filename=tar_name, timestamp=timestamp)
        trainer.eval_logger.info(ref_list)

        if conf.pretrained_baseline_path:
            trainer.load_pretrain_model(conf.pretrained_baseline_path)

        trainer.set_baseline_img()

        # train Dn network
        trainer.freeze_network(trainer.en_model)
        trainer.freeze_network(trainer.sr_model)
        if conf.pretrained_lr_path:
            lr_model_path = lr_model_path_list[idx]
            trainer.load_dn_model(lr_model_path)
            trainer.logger.info(f'Load model {lr_model_path} for {os.path.basename(target_path)}')
        else:
            conf.train_iters = origin_iter
            # TODO: [DOE] add downsample regularization
            # multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, ref_list)
            dn_train_dataloader = gen_dn_train_dataloader(conf, target_path)
            set_loader_seed(dn_train_dataloader, conf.random_seed)
            # TODO: [DOE] add downsample regularization
            # for ind, data_dict in enumerate(multi_train_dataloader):
            for ind, data_dict in enumerate(dn_train_dataloader):
                # TODO: [DOE] add downsample regularization
                # trainer.set_input(data_dict)
                trainer.set_dn_input(data_dict)
                # trainer.save_input_img()
                target_hr_base = trainer.get_target_baseline_result()
                # TODO: [DOE] add downsample regularization
                trainer.start_train_dn(target_hr_base)
                # trainer.start_train_dn_v2(target_hr_base)
                tb_logger.flush()

            trainer.save_model()

            # load best dn model from previous training
            time.sleep(1)
            best_dn_model_path = os.path.join(trainer.save_path, f'model_dn_{trainer.filename}_best.pt')
            if os.path.exists(best_dn_model_path):
                trainer.load_dn_model(best_dn_model_path)

        # save ref_img
        save_ref_img(ref_list, trainer.save_path)

        best_sr_model_path = os.path.join(trainer.save_path, f'model_sr_{trainer.filename}_best.pt')
        base_sr_model_path = os.path.join(trainer.save_path, f'model_sr_{trainer.filename}_base.pt')
        conf.ref_count = 1

        # TODO: set baseline threshold
        trainer.en_model.eval()
        trainer.sr_model.eval()
        trainer.dn_evaluation(is_dn=False)
        trainer.sr_evaluation()
        trainer.cal_whole_image_loss(is_dn=False)

        # trainer.init_ref_img()

        # train encoder
        # unfreeze encoder
        trainer.unfreeze_network(trainer.en_model)
        for i, ref_path in enumerate(ref_list):
            if os.path.exists(best_sr_model_path):
                trainer.load_sr_model(best_sr_model_path)
            elif os.path.exists(best_sr_model_path):
                trainer.load_sr_model(base_sr_model_path)

            # train SR
            conf.train_iters = conf.train_encoder_iters
            trainer.init_optimizer(is_dn=False, is_en=True, is_up=False)
            trainer.init_scheduler(is_dn=False, is_en=True, is_up=False)

            # init SR dataloader
            multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, [ref_path])

            for ind, data_dict in enumerate(multi_train_dataloader):
                trainer.set_input(data_dict)
                target_hr_base = trainer.get_target_baseline_result()
                target_hr_dr = trainer.get_target_baseline_representation()
                trainer.start_train_up(target_hr_dr, target_hr_base, matrix=conf.lrs_matrix)
                tb_logger.flush()
                trainer.flush_log()

        # train sr network
        # unfreeze SR model
        trainer.unfreeze_network(trainer.sr_model)
        for i, ref_path in enumerate(ref_list):
            if os.path.exists(best_sr_model_path):
                trainer.load_sr_model(best_sr_model_path)
            elif os.path.exists(best_sr_model_path):
                trainer.load_sr_model(base_sr_model_path)

            # train SR
            conf.train_iters = conf.train_sr_iters
            trainer.init_optimizer(is_dn=False, is_en=False, is_up=True)
            trainer.init_scheduler(is_dn=False, is_en=False, is_up=True)

            # init SR dataloader
            multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, [ref_path])

            for ind, data_dict in enumerate(multi_train_dataloader):
                trainer.set_input(data_dict)
                target_hr_base = trainer.get_target_baseline_result()
                target_hr_dr = trainer.get_target_baseline_representation()
                trainer.start_train_up(target_hr_dr, target_hr_base, sr=True, matrix=conf.lrs_matrix)
                tb_logger.flush()
                trainer.flush_log()

        trainer.finish()


def main():
    # this is for training with gt kernel & gt ref image
    # configuration initialize
    # opt = Options()
    opt = JsonOptions()

    conf = opt.get_config()
    if conf is None:
        print('config is None')
    # TODO: [DOE] set random seed
    set_seed(conf.random_seed)

    if conf.ref_select_path:
        select_ref_list = sorted(glob.glob(os.path.join(conf.ref_select_path, "*")))
        select_ref_list = sorted(select_ref_list, key=lambda x: int(os.path.basename(x).split('_')[1]))
        select_ref_list = ['_'.join(os.path.basename(ref).split('_')[:2]) + '.png' for ref in select_ref_list]
        print(','.join(select_ref_list))
    # create timestamp and logger
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    create_train_logger(timestamp, conf.train_log)
    tb_logger = create_TBlogger(conf)

    # dump training configuration to files
    dump_training_settings(conf, timestamp)

    # setup training data folders
    ref_path_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.ref_dir, "*")))
    target_path_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.target_dir, "*.png")))

    ref_gt_path_list = []
    if conf.ref_gt_dir:
        ref_gt_path_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.ref_gt_dir, "*")))

    target_gt_path_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.target_gt_dir, "*.png")))

    # TODO: key sort function may be modified by datasets
    target_path_list = sorted(target_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
    target_gt_path_list = sorted(target_gt_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))

    # this is for specific reference
    if not conf.ref_pool:
        ref_path_list = sorted(ref_path_list, key=lambda x: int(os.path.basename(x)))

    # This is only for down sample
    kernel_gt_path_list = ''
    if conf.kernel_gt_dir:
        kernel_gt_path_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.kernel_gt_dir, "*.mat")))
        kernel_gt_path_list = sorted(kernel_gt_path_list, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

    # this is for customized training
    if conf.target_count and len(target_path_list) > conf.target_ind:
        target_path_list = target_path_list[conf.target_ind:conf.target_ind+conf.target_count]
        target_gt_path_list = target_gt_path_list[conf.target_ind:conf.target_ind+conf.target_count]
        if conf.kernel_gt_dir:
            kernel_gt_path_list = kernel_gt_path_list[conf.target_ind:conf.target_ind+conf.target_count]
        if not conf.ref_pool:
            ref_path_list = ref_path_list[conf.target_ind:conf.target_ind+conf.target_count]
        print(f"target len:{len(target_path_list)}")

    # start to train
    train(conf, tb_logger, target_path_list, ref_path_list,
          timestamp=timestamp, gt_tar=target_gt_path_list, gt_ref=ref_gt_path_list, gt_ker=kernel_gt_path_list)
    tb_logger.close()


if __name__ == '__main__':
    main()
