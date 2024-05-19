from options import Options, JsonOptions
from utils.utils import create_TBlogger, create_train_logger, dump_training_settings, set_loader_seed, set_seed
from datetime import datetime
from utils.img_utils import sample_ref_by_color_space, kernel_processing, save_ref_img

from data.data import gen_test_dataloader, gen_train_dataloader_adaptive, gen_dn_train_dataloader
# from trainer import RDSRTrainer
from trainer.rdsrdisctrainerv83 import RDSRDiscTrainerV83
from utils.utils import ref_preprocessing, selection_mode

import os
import glob
import random
import time
import torch
import numpy as np
import pickle
import shutil

torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True)
torch.manual_seed(24)
random.seed(24)
np.random.seed(24)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# import multiprocessing as mp
# mp.get_start_method()


def train(conf, tb_logger, tar_path_list, ref_path_list, timestamp='', gt_tar='', gt_ker='', gt_ref='', save_path=''):
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    if len(gt_ref) == 0:
        gt_ref = ''

    # this function is for target & ref select path
    tar_path_list, ref_path_list, gt_tar, gt_ker = selection_mode(conf, tar_path_list, ref_path_list, gt_tar, gt_ker)

    # Add LR pretrain model
    if conf.pretrained_lr_path:
        try:
            lr_model_path_list = glob.glob(os.path.join(conf.pretrained_lr_path, "*.pt"))
            lr_model_path_list = sorted(lr_model_path_list, key=lambda x: int(x.split("_")[-2]))
            lr_model_path_list = lr_model_path_list[conf.target_ind:conf.target_ind+conf.target_count]
            print(lr_model_path_list)
        except:
            pass

    origin_iter = conf.train_iters
    origin_ref_cnt = conf.ref_count
    origin_evaluate_iters = conf.evaluate_iters

    #create folder for dr
    root_dr =  f'dr_result/{conf.exp_name}target_ind_{conf.target_ind}_target_count_{conf.target_count}'
    shutil.rmtree(root_dr, ignore_errors=True)
    os.makedirs(root_dr)

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

        # apply ref_count
        # ref_list = ref_list[0:conf.ref_count]

        # setup test dataset
        test_dataloader = gen_test_dataloader(
            conf, target_path, ref_list, tar_gt=target_gt_path, ref_gt=ref_gt_list)

        # for _ in range(10):
        #     print(next(iter(test_dataloader)))

        tar, ext = os.path.splitext(os.path.basename(target_path))
        tar_name = '_'.join(tar.split('_')[:2])

        # TODO: to check trainer process
        trainer = RDSRDiscTrainerV83(conf, tb_logger, test_dataloader, kernel=ker_gt_path, filename=tar_name, timestamp=timestamp)
        trainer.eval_logger.info(ref_list)

        if conf.pretrained_baseline_path:
            trainer.load_pretrain_model(conf.pretrained_baseline_path)

        trainer.set_baseline_img()

        # train Dn network
        trainer.freeze_network(trainer.en_model)
        trainer.freeze_network(trainer.sr_model)
        if conf.pretrained_lr_path:
            # lr_model_path = lr_model_path_list[idx]
            lr_model_path = './dn_model_3000iter_64bs.pt' ######
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
            conf.evaluate_iters = origin_evaluate_iters
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

            # path = './dn_model_3000iter_64bs.pt'
            # torch.save(trainer.dn_model.state_dict(), path)

            # trainer.dn_model.load_state_dict(torch.load(path), strict=False)

        # ================= Phu tune DN ==================
        # path = './dn_model_aftertuning.pt'
        # trainer.load_dn_model(path)
        # trainer.logger.info(f'{path}') 

        # print('tuning DN ... /n')
        # for i, ref_path in enumerate(ref_list):
        #     # init SR dataloader
        #     trainer.list_targetLR_dr = []
        #     trainer.list_refLR_dr = []
        #     # conf.ref_count = 1
        #     conf.train_iters = 20
        #     conf.evaluate_iters = conf.scale_iters
        #     multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, [ref_path])
        #     trainer.init_optimizer(is_dn=False, is_en=False, is_up=False, is_dn_2=True)
        #     trainer.init_scheduler(is_dn=True, is_en=False, is_up=False)

        #     # training DN with Ltarget + Lreg
        #     for epoch in range(20):
        #         for ind, data_dict in enumerate(multi_train_dataloader):
        #             trainer.list_targetLR_dr = []
        #             trainer.list_refLR_dr = []
        #             trainer.set_input(data_dict)
        #             target_hr_base = trainer.get_target_baseline_result()
                    
        #             trainer.start_train_dn_with_Lreg(target_hr_base)
        #             tb_logger.flush()
        #             trainer.flush_log()
        #         # with open(f'{root_dr}/{idx}_LR_{i}_HR_list_targetLR_dr.pickle', 'wb') as w:
        #         #     pickle.dump(trainer.list_targetLR_dr, w)
                
        #         # with open(f'{root_dr}/{idx}_LR_{i}_HR_list_refLR_dr.pickle', 'wb') as w:
        #         #     pickle.dump(trainer.list_refLR_dr, w)
                    
                    
        #             if epoch % 100 == 0:
        #                 with open(f'{root_dr}/{epoch}_epoch_{idx}_LR_{ind}_HR_list_targetLR_dr.pickle', 'wb') as w:
        #                     pickle.dump(trainer.list_targetLR_dr, w)
                        
        #                 with open(f'{root_dr}/{epoch}_epoch_{idx}_LR_{ind}_HR_list_refLR_dr.pickle', 'wb') as w:
        #                     pickle.dump(trainer.list_refLR_dr, w)

        

            
        trainer.save_path =   save_path   
        trainer.save_model()
        torch.cuda.empty_cache()

        # load best dn model from previous training
        time.sleep(1)
        best_dn_model_path = os.path.join(trainer.save_path, f'model_dn_{trainer.filename}_best.pt')

        best_dn_model_path = os.path.join(trainer.save_path, f'model_dn_fintune_best.pt')
        trainer.logger.info(f'load {best_dn_model_path} after finetune DN')
        if os.path.exists(best_dn_model_path):
            trainer.load_dn_model(best_dn_model_path)

        # save ref_img
        # save_ref_img(ref_list, trainer.save_path)

        best_sr_model_path = os.path.join(trainer.save_path, f'model_sr_{trainer.filename}_best.pt')
        base_sr_model_path = os.path.join(trainer.save_path, f'model_sr_{trainer.filename}_base.pt')
        # conf.ref_count = 1

        # TODO: set baseline threshold
        trainer.en_model.eval()
        trainer.sr_model.eval()
        trainer.dn_evaluation(is_dn=False)
        trainer.sr_evaluation()
        trainer.cal_whole_image_loss(is_dn=False)
        # trainer.freeze_network(trainer.dn_model)
        # trainer.unfreeze_network(trainer.en_model)

        # train encoder
        # unfreeze encoder
        trainer.logger.info(f'tuning en network ... at {trainer.iter}')
        trainer.unfreeze_network(trainer.en_model)
        for i, ref_path in enumerate(ref_list):
            if os.path.exists(best_sr_model_path):
                trainer.load_sr_model(best_sr_model_path)
            elif os.path.exists(best_sr_model_path):
                trainer.load_sr_model(base_sr_model_path)

            # train encoder
            conf.train_iters = conf.train_encoder_iters
            conf.evaluate_iters = origin_evaluate_iters
            # conf.batch_size = 32
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
        trainer.logger.info(f'tuning sr network ... at {trainer.iter}')

        trainer.unfreeze_network(trainer.sr_model)
        for i, ref_path in enumerate(ref_list):
            if os.path.exists(best_sr_model_path):
                trainer.load_sr_model(best_sr_model_path)
            elif os.path.exists(best_sr_model_path):
                trainer.load_sr_model(base_sr_model_path)

            # train SR
            conf.train_iters = conf.train_sr_iters
            conf.evaluate_iters = conf.scale_iters
            # conf.batch_size = 32
            trainer.init_optimizer(is_dn=False, is_en=False, is_up=True)
            trainer.init_scheduler(is_dn=False, is_en=False, is_up=True)

            # init SR dataloader
            multi_train_dataloader = gen_train_dataloader_adaptive(conf, target_path, [ref_path])

            for ind, data_dict in enumerate(multi_train_dataloader):
                trainer.set_input(data_dict)
                target_hr_base = trainer.get_target_baseline_result()
                target_hr_dr = trainer.get_target_baseline_representation()
                # trainer.start_train_up_addLRloss(target_hr_dr, target_hr_base, sr=True, matrix=conf.lrs_matrix)
                trainer.start_train_up(target_hr_dr, target_hr_base, sr=True, matrix=conf.lrs_matrix)

                tb_logger.flush()
                trainer.flush_log()
        # trainer.dn_evaluation(is_dn=False)
        trainer.finish()
        # break


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
        select_ref_list = sorted(glob.glob(os.path.join(conf.datasets_dir, conf.ref_select_path, "*")))
        select_ref_list = sorted(select_ref_list, key=lambda x: int(os.path.basename(x).split('_')[1]))
        select_ref_list = ['_'.join(os.path.basename(ref).split('_')[:2]) + '.png' for ref in select_ref_list]
        print(','.join(select_ref_list))
    # create timestamp and logger
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    savePath = create_train_logger(timestamp, conf.train_log, conf.exp_name)
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
          timestamp=timestamp, gt_tar=target_gt_path_list, gt_ref=ref_gt_path_list, gt_ker=kernel_gt_path_list, save_path=savePath)
    tb_logger.close()


if __name__ == '__main__':
    main()