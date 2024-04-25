import os
import tqdm
import torch
from options1 import options, JsonOptions
from data import create_dataset
from data1 import gen_target_ref_test_data, gen_target_ref_train_data, gen_target_ref_train_data_v2
from DualSR20 import DualSR
from networks1 import make_dasr_network
from learner import Learner
import util
import glob

# use DualSR2
# add linear initialization
# remove representation on down sampling


def ref_selection(conf, ref_path_list, tar_path_list):
    random.seed(conf.ref_random)
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

    return ref_all_list


def train_and_eval(conf, tar_path, ref_list, dasr_net=None):
    encoder_network = dasr_net.E

    test_loader = gen_target_ref_test_data(conf, tar_path, conf.gt_path, ref_list)
    model = DualSR(conf, test_loader=test_loader, encoder_network=dasr_net.E)
    model.logger.info(ref_list)
    # dataloader = create_dataset(conf)
    # model.eval_whole_img()
    train_loader = gen_target_ref_train_data(conf, tar_path, ref_list)
    # train_loader = gen_target_ref_train_data_v2(conf, tar_path, ref_list)
    learner = Learner(model)
    print('*' * 60 + '\nTraining started ...')
    # model.eval_ref_w_baseline()

    # for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    # model.save_test_loader_image()
    for iteration, data in enumerate(train_loader):
    # for iteration, data in enumerate(dataloader):
        model.train_v2(data)
        # learner.update(iteration, model)

    model.finish()
    # model.eval(is_final=True)


def main():
    # opt = options()
    opt = JsonOptions()
    conf = opt.load_config()
    if conf is None:
        print('config is None')
        return
    # Run DualSR on all images in the input directory

    util.dump_training_settings(conf)
    if not conf.div2k:
        ref_path_list = sorted(glob.glob(os.path.join(conf.ref_dir, "*")))
        target_path_list = sorted(glob.glob(os.path.join(conf.input_dir, "*.png")))
        target_path_list = sorted(target_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
        target_path_list = target_path_list[conf.target_ind:conf.target_ind + conf.target_count]
        if conf.ref_select_path:
            ref_all_list = ref_selection(conf, ref_path_list, target_path_list)
    else:
        target_path_list = sorted(glob.glob(os.path.join(conf.input_dir, "*.png")))
        ref_path_list = sorted(glob.glob(os.path.join(conf.ref_dir, "*.png")))
        ref_path_list = sorted(ref_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
        # target_path_list = sorted(target_path_list, key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
        if conf.target_select_path:
            with open(conf.target_select_path, 'r') as fp:
                target_list_str = fp.read()
                tar_list = target_list_str.split(',')
            # (x == big_foobar for x in foobars)
            target_path_list = [tar for tar in target_path_list if any(t in tar for t in tar_list)]
        else:
            target_path_list = target_path_list[conf.target_ind:conf.target_ind + conf.target_count]
        # ref_all_list = ref_preprocessing(conf, ref_path_list, target_path_list)

    dasr_net = make_dasr_network(conf)

    # for img_name in os.listdir(opt.conf.input_dir):
    for idx, img_name in enumerate(target_path_list):
        # im_1, 3000, 30.70463
        if not conf.div2k:
            conf = opt.get_config(img_name)
            if conf.ref_select_path:
                ref_list = ref_all_list[idx]
                ref_list = ref_list[:conf.ref_count]
            else:
                ref_list = util.ref_preprocessing(conf, ref_path_list)
        else:
            conf = opt.get_div2k_config(img_name)
            ref_list = util.ref_preprocessing(conf, ref_path_list)

        dasr_net.load_state_dict(torch.load(conf.pretrained_encoder_path), strict=False)
        train_and_eval(conf, img_name, ref_list, dasr_net=dasr_net)


if __name__ == '__main__':
    main()
