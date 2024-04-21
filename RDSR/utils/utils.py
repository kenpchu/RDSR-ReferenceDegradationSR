import os
import json
import logging
import sys
import torch
import numpy as np
import random
import glob

from datetime import datetime
from tensorboardX import SummaryWriter
from utils.img_utils import sample_ref_by_color_space


def create_TBlogger(conf):
    if not os.path.exists('../tb_loggers'):
        os.mkdir('../tb_loggers')
    if conf.tb_name != 'TIME':
        path = 'tb_loggers/{}'.format(conf.tb_name)
    else:
        dir = datetime.now().strftime('%m_%d_%Y_%H%M%S')
        path = 'tb_loggers/{}'.format(dir)
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path)
    return writer


def create_train_logger(timestamp, train_log_name):
    if not os.path.exists(f'{train_log_name}/{timestamp}'):
        os.makedirs(f'{train_log_name}/{timestamp}')
    main_logger = logging.getLogger(f"{timestamp}")
    f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/main_{timestamp}.log")
    f_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    main_logger.setLevel(logging.DEBUG)
    main_logger.addHandler(f_handler)
    main_logger.addHandler(stream_handler)

    eval_logger = logging.getLogger(f"eval")
    eval_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/eval_{timestamp}.csv")
    eval_f_handler.setLevel(logging.DEBUG)
    eval_logger.setLevel(logging.DEBUG)
    eval_logger.addHandler(eval_f_handler)

    loss_logger = logging.getLogger(f"loss")
    loss_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/loss_{timestamp}.csv")
    loss_f_handler.setLevel(logging.DEBUG)
    loss_logger.setLevel(logging.DEBUG)
    loss_logger.addHandler(loss_f_handler)

    lr_logger = logging.getLogger(f"lr")
    lr_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/lr_{timestamp}.csv")
    lr_f_handler.setLevel(logging.DEBUG)
    lr_logger.setLevel(logging.DEBUG)
    lr_logger.addHandler(lr_f_handler)

    lr_logger.info(f'iteration, en_lr, sr_lr, dn_lr, {timestamp}')
    loss_logger.info(f'iteration, target_sr_loss, target_sr_vgg_loss, tar_hf_loss, ref_loss, ref_vgg_loss, loss_tar_lr,'
                     f' loss_tar_lr_vgg, loss_interpo, loss_tv, loss_ref_hf, total_loss, loss_ref_gv, total_loss, {timestamp}')
    eval_logger.info(f'iteration, target_hr_psnr, tar_rec_lr_psnr, ref_rec_hr_psnr, ref_lr_psnr, {timestamp}')


def create_train_logger2(timestamp, train_log_name):
    if not os.path.exists(f'{train_log_name}/{timestamp}'):
        os.makedirs(f'{train_log_name}/{timestamp}')
    main_logger = logging.getLogger(f"{timestamp}")
    f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/main_{timestamp}.log")
    f_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    main_logger.setLevel(logging.DEBUG)
    main_logger.addHandler(f_handler)
    main_logger.addHandler(stream_handler)

    eval_logger = logging.getLogger(f"eval")
    eval_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/eval_{timestamp}.csv")
    eval_f_handler.setLevel(logging.DEBUG)
    eval_logger.setLevel(logging.DEBUG)
    eval_logger.addHandler(eval_f_handler)

    loss_logger = logging.getLogger(f"loss")
    loss_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/loss_{timestamp}.csv")
    loss_f_handler.setLevel(logging.DEBUG)
    loss_logger.setLevel(logging.DEBUG)
    loss_logger.addHandler(loss_f_handler)

    lr_logger = logging.getLogger(f"lr")
    lr_f_handler = logging.FileHandler(f"{train_log_name}/{timestamp}/lr_{timestamp}.csv")
    lr_f_handler.setLevel(logging.DEBUG)
    lr_logger.setLevel(logging.DEBUG)
    lr_logger.addHandler(lr_f_handler)

    lr_logger.info(f'iteration, sr_lr, dn_lr, en_lr, {timestamp}')
    loss_logger.info(
        f'iteration, tar_w_loss, tar_vgg_w_loss, ref_w_loss, ref_w_regular_loss, ref_vgg_w_loss, {timestamp}')
    eval_logger.info(f'iteration, tar_hr_w_psnr, tar_lr_w_psnr, tar_lr_gt_w_psnr, ref_hr_psnr, {timestamp}')


def dump_training_settings(conf, timestamp):
    # dump config to dictionary
    train_folder = os.path.join(conf.train_log, timestamp)
    conf_dict = dict()
    for arg in vars(conf):
        conf_dict[arg] = getattr(conf, arg)

    with open(os.path.join(train_folder, 'config.json'), 'w') as conf_fp:
        json.dump(conf_dict, conf_fp, indent=4)


def set_seed(seed=0):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_loader_seed(loader, seed):
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


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
