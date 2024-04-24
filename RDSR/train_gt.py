from options import Options, JsonOptions
from utils.utils import create_TBlogger, create_train_logger, dump_training_settings
from datetime import datetime
from utils.img_utils import sample_ref_by_color_space, kernel_processing, save_ref_img

from data.data import gen_test_dataloader, gen_train_dataloader_adaptive
# from trainer import RDSRTrainer
from trainer.rdsrgttrainer import RDSRGtTrainer

import os
import glob
import random


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


def train(conf, tb_logger, tar_path_list, ref_path_list, timestamp='', gt_tar='', gt_ref='', gt_ker=''):
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    if len(gt_ref) == 0:
        gt_ref = ''

    # Add LR pretrain model
    if conf.pretrained_lr_path:
        lr_model_path_list = glob.glob(os.path.join(conf.pretrained_lr_path, "*.pt"))
        lr_model_path_list = sorted(lr_model_path_list, key=lambda x: int(x.split("_")[-2]))
        print(lr_model_path_list)

    origin_iter = conf.train_iters
    for idx, target_path in enumerate(tar_path_list):
        # set gt target & gt kernel
        if gt_tar:
            target_gt_path = gt_tar[idx]

        ker_gt_path = ''
        if gt_ker:
            ker_gt_path = gt_ker[idx]

        # set reference images
        ref_list, ref_gt_list = ref_preprocessing(conf, ref_path_list, idx, tar_path_list, gt_ref=gt_ref)

        # setup test dataset
        test_dataloader = gen_test_dataloader(
            conf, target_path, ref_list, tar_gt=target_gt_path, ref_gt=ref_gt_list)

        # for _ in range(10):
        #     print(next(iter(test_dataloader)))

        tar, ext = os.path.splitext(os.path.basename(target_path))
        tar_name = '_'.join(tar.split('_')[:2])

        conf.train_iters = origin_iter
        # TODO: to check trainer process
        trainer = RDSRGtTrainer(conf, tb_logger, test_dataloader, kernel=ker_gt_path, filename=tar_name, timestamp=timestamp)
        trainer.eval_logger.info(ref_list)

        # save ref_img
        save_ref_img(ref_list, trainer.save_path)

        if conf.pretrained_baseline_path:
            trainer.load_pretrain_model(conf.pretrained_baseline_path)

        conf.train_iters = conf.train_encoder_iters + conf.train_sr_iters
        multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, ref_list)

        trainer.set_baseline_img()
        trainer.unfreeze_network(trainer.en_model)
        trainer.unfreeze_network(trainer.sr_model)
        for ind, data_dict in enumerate(multi_train_dataloader):
            trainer.set_input(data_dict)
            target_hr_base = trainer.get_target_baseline_result()
            target_hr_dr = trainer.get_target_baseline_representation()
            if ind < conf.train_encoder_iters:
                trainer.start_train_rdsr(target_hr_dr, target_hr_base, matrix=conf.lrs_matrix)
            else:
                trainer.start_train_rdsr(target_hr_dr, target_hr_base, sr=True, matrix=conf.lrs_matrix)
            tb_logger.flush()
            trainer.flush_log()

        trainer.finish()


def main():
    # this is for training with gt kernel & gt ref image
    # configuration initialize
    # opt = Options()
    opt = JsonOptions()
    conf = opt.get_config()

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
