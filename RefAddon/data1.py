from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from scipy.signal import convolve2d
from util import read_image, im2tensor, create_gradient_map, create_probability_map, nn_interpolation
from util import np2tensor
from imresize import imresize

import math
import numpy as np
import random


# The preprocessing of dataloader is followed as DualSR

def gen_target_ref_test_data(conf, target_path, tar_gt, ref_path_list):
    dataset = TarRefTestDatasets(conf, target_path, tar_gt, ref_path_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def gen_target_ref_train_data(conf, target_path, ref_path_list):
    dataset = TarRefTrainDataGenerator(conf, target_path, ref_path_list)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)

    return dataloader


def gen_target_ref_train_data_v2(conf, target_path, ref_path_list):
    dataset = TarRefTrainDataGeneratorV2(conf, target_path, ref_path_list)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)

    return dataloader


class TarRefTestDatasets(Dataset):

    def __init__(self, conf, tar_path, tar_gt_path, ref_path_list):
        np.random.seed(0)
        self.conf = conf
        self.conf.scale = conf.scale_factor
        self.target_image = read_image(tar_path) / 255.
        self.target_gt_image = read_image(tar_gt_path) / 255.
        self.ref_img_list = []
        if ref_path_list:
            for ref_path in ref_path_list:
                self.ref_img_list.append(read_image(ref_path) / 255.)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        tar_img = np2tensor(self.target_image)
        tar_gt = np2tensor(self.target_gt_image)
        tar_img_bq = imresize(im=self.target_image, scale_factor=self.conf.scale_factor, kernel='cubic')
        tar_img_bq_t = np2tensor(tar_img_bq)
        ref_img_list = [np2tensor(ref) for ref in self.ref_img_list]
        ref_bq_img_list = []

        for ref in self.ref_img_list:
            ref_bq_dn = imresize(im=ref, scale_factor=1.0 / self.conf.scale, kernel='cubic')
            ref_bq_dn_up = imresize(im=ref_bq_dn, scale_factor=self.conf.scale, kernel='cubic')
            ref_bq_img_list.append(np2tensor(ref_bq_dn_up))

        data = {
            'Target_Img': tar_img,
            'Target_Gt': tar_gt,
            'Target_Img_Bq': tar_img_bq_t,
            'Ref_Imgs': ref_img_list,
            'Ref_Bq_Imgs': ref_bq_img_list,
        }

        return data


class TarRefTrainDataGenerator(Dataset):

    def __init__(self, conf, tar_path, ref_path_list):
        np.random.seed(0)
        self.conf = conf
        self.conf.scale = conf.scale_factor
        self.conf.train_iters = conf.num_iters

        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = int(conf.input_crop_size * (1/conf.scale))

        self.target_image = read_image(tar_path) / 255.
        self.target_shave_edges(conf.scale, False)

        self.target_in_rows, self.target_in_cols = self.target_image.shape[0:2]
        self.target_crop_indices_for_g, self.target_crop_indices_for_d = self.make_target_list_of_crop_indices()

        self.ref_path_list = ref_path_list
        self.ref_imgs = []
        self.ref_imgs_size = []
        self.ref_crop_imgs = []

        for ref_img_path in ref_path_list:
            tmp_ref_img = read_image(ref_img_path) / 255.
            tmp_ref_img = self.ref_shave_edges(conf.scale, False, tmp_ref_img)
            tmp_in_rows, tmp_in_cols = tmp_ref_img.shape[0:2]
            g_crop_indices, d_crop_indices = self.make_list_of_crop_indices_m(tmp_ref_img)
            self.ref_imgs.append(tmp_ref_img)
            self.ref_crop_imgs.append((g_crop_indices, d_crop_indices))
            self.ref_imgs_size.append((tmp_in_rows, tmp_in_cols))

    @staticmethod
    def ref_shave_edges(scale_factor, real_image, tmp_ref_img):
        if not real_image:
            tmp_ref_img = tmp_ref_img[10:-10, 10:-10, :]
        sf = scale_factor
        shape = tmp_ref_img.shape
        tmp_ref_img = tmp_ref_img[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else tmp_ref_img
        tmp_ref_img = tmp_ref_img[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else tmp_ref_img
        return tmp_ref_img

    def target_shave_edges(self, scale, real_image):
        if not real_image:
            self.target_image = self.target_image[10:-10, 10:-10, :]
        shape = self.target_image.shape
        self.target_image = self.target_image[:-(shape[0] % scale), :, :] if shape[0] % scale > 0 else self.target_image
        self.target_image = self.target_image[:, :-(shape[1] % scale), :] if shape[1] % scale > 0 else self.target_image

    def make_target_list_of_crop_indices(self):
        iterations = self.conf.train_iters * self.conf.batch_size
        prob_map_big, prob_map_sml = self.create_target_prob_maps(1/self.conf.scale)

        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def make_list_of_crop_indices_m(self, img):
        iterations = math.ceil(self.conf.train_iters * self.conf.batch_size / self.conf.ref_count * 1.)
        prob_map_big, prob_map_sml = self.create_prob_maps_m(1/self.conf.scale, img)

        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps_m(self, scale_factor, img):
        loss_map_big = create_gradient_map(img)
        loss_map_sml = create_gradient_map(imresize(im=img, scale_factor=scale_factor, kernel='cubic'))
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def create_target_prob_maps(self, scale_factor):
        loss_map_big = create_gradient_map(self.target_image)
        loss_map_sml = create_gradient_map(imresize(im=self.target_image, scale_factor=scale_factor, kernel='cubic'))
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def get_target_top_left(self, size, for_g, idx):
        center = self.target_crop_indices_for_g[idx] if for_g else self.target_crop_indices_for_d[idx]
        row, col = int(center / self.target_in_cols), center % self.target_in_cols
        top, left = min(max(0, row - size // 2), self.target_in_rows - size), min(max(0, col - size // 2), self.target_in_cols - size)
        return top - top % 2, left - left % 2

    def ref_get_top_left(self, size, for_g, img_idx, iter_idx):
        # print(img_idx)
        # print(iter_idx)
        center = self.ref_crop_imgs[img_idx][0][iter_idx] if for_g else self.ref_crop_imgs[img_idx][1][iter_idx]
        in_cols = self.ref_imgs_size[img_idx][1]
        in_rows = self.ref_imgs_size[img_idx][0]

        row, col = int(center / in_cols), center % in_cols
        top, left = min(max(0, row - size // 2), in_rows - size), min(max(0, col - size // 2), in_cols - size)
        return top - top % 2, left - left % 2

    def target_next_crop(self, for_g, idx):
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_target_top_left(size, for_g, idx)
        crop_im = self.target_image[top:top + size, left:left + size, :]
        return crop_im

    def ref_next_crop(self, img_idx, iter_idx, for_g):
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.ref_get_top_left(size, for_g, img_idx, iter_idx)
        crop_im = self.ref_imgs[img_idx][top:top + size, left:left + size, :]
        return crop_im

    def __len__(self):
        # return self.conf.train_iters * self.conf.batch_size + self.conf.train_sr_iters * self.conf.batch_size
        return self.conf.train_iters * self.conf.batch_size

    def __getitem__(self, idx):
        if self.conf.ref_continue_mode:
            unit_count = math.ceil(self.conf.train_iters * self.conf.batch_size / self.conf.ref_count * 1.)
            img_idx = idx // unit_count
            iter_idx = idx - img_idx * unit_count
        else:
            img_idx = idx % self.conf.ref_count
            iter_idx = (idx - img_idx) // self.conf.ref_count

        # target
        t_g_in = self.target_next_crop(for_g=True, idx=idx)
        t_d_in = self.target_next_crop(for_g=False, idx=idx)
        t_d_bq_up = imresize(im=t_d_in, scale_factor=self.conf.scale, kernel='cubic')
        # ref
        ref_g_in = self.ref_next_crop(img_idx, iter_idx, for_g=True)
        ref_g_bq_dn = imresize(im=ref_g_in, scale_factor=1.0/self.conf.scale, kernel='cubic')
        ref_g_bq_dn_up = imresize(im=ref_g_bq_dn, scale_factor=self.conf.scale, kernel='cubic')
        ref_d_in = self.ref_next_crop(img_idx, iter_idx, for_g=False)
        ref_d_bq_up = imresize(im=ref_d_in, scale_factor=self.conf.scale, kernel='cubic')

        # return {'tar_big_patch': np2tensor(t_g_in),
        #         'tar_small_patch': np2tensor(t_d_in),
        #         'tar_small_bq_up': np2tensor(t_d_bq_up),
        #         'ref_big_patch': np2tensor(ref_g_in),
        #         'ref_small_patch': np2tensor(ref_d_in),
        #         'ref_small_bq_up': np2tensor(ref_d_bq_up)}

        return {'HR': np2tensor(t_g_in),
                'LR': np2tensor(t_d_in),
                'LR_bicubic': np2tensor(t_d_bq_up),
                'ref_big_patch': np2tensor(ref_g_in),
                'ref_big_bq_dn_up': np2tensor(ref_g_bq_dn_up),
                'ref_small_patch': np2tensor(ref_d_in),
                'ref_small_bq_up': np2tensor(ref_d_bq_up)}


class TarRefTrainDataGeneratorV2(Dataset):

    def __init__(self, conf, tar_path, ref_path_list):
        np.random.seed(0)
        self.conf = conf
        self.conf.scale = conf.scale_factor
        self.conf.train_iters = conf.num_iters

        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = int(conf.input_crop_size * (1/conf.scale))

        self.target_image = read_image(tar_path) / 255.
        self.target_shave_edges(conf.scale, False)

        self.target_in_rows, self.target_in_cols = self.target_image.shape[0:2]
        self.target_crop_indices_for_g, self.target_crop_indices_for_d = self.make_target_list_of_crop_indices()

        self.ref_path_list = ref_path_list
        self.ref_imgs = []
        self.ref_imgs_size = []
        self.ref_crop_imgs = []

        for ref_img_path in ref_path_list:
            tmp_ref_img = read_image(ref_img_path) / 255.
            tmp_ref_img = self.ref_shave_edges(conf.scale, False, tmp_ref_img)
            tmp_in_rows, tmp_in_cols = tmp_ref_img.shape[0:2]
            g_crop_indices, d_crop_indices = self.make_list_of_crop_indices_m(tmp_ref_img)
            self.ref_imgs.append(tmp_ref_img)
            self.ref_crop_imgs.append((g_crop_indices, d_crop_indices))
            self.ref_imgs_size.append((tmp_in_rows, tmp_in_cols))

    @staticmethod
    def ref_shave_edges(scale_factor, real_image, tmp_ref_img):
        if not real_image:
            tmp_ref_img = tmp_ref_img[10:-10, 10:-10, :]
        sf = scale_factor
        shape = tmp_ref_img.shape
        tmp_ref_img = tmp_ref_img[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else tmp_ref_img
        tmp_ref_img = tmp_ref_img[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else tmp_ref_img
        return tmp_ref_img

    def target_shave_edges(self, scale, real_image):
        if not real_image:
            self.target_image = self.target_image[10:-10, 10:-10, :]
        shape = self.target_image.shape
        self.target_image = self.target_image[:-(shape[0] % scale), :, :] if shape[0] % scale > 0 else self.target_image
        self.target_image = self.target_image[:, :-(shape[1] % scale), :] if shape[1] % scale > 0 else self.target_image

    def make_target_list_of_crop_indices(self):
        iterations = self.conf.train_iters * self.conf.batch_size
        prob_map_big, prob_map_sml = self.create_target_prob_maps(1/self.conf.scale)

        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def make_list_of_crop_indices_m(self, img):
        iterations = math.ceil(self.conf.train_iters * self.conf.batch_size / self.conf.ref_count * 1.)
        prob_map_big, prob_map_sml = self.create_prob_maps_m(1/self.conf.scale, img)

        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps_m(self, scale_factor, img):
        loss_map_big = create_gradient_map(img)
        loss_map_sml = create_gradient_map(imresize(im=img, scale_factor=scale_factor, kernel='cubic'))
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def create_target_prob_maps(self, scale_factor):
        loss_map_big = create_gradient_map(self.target_image)
        loss_map_sml = create_gradient_map(imresize(im=self.target_image, scale_factor=scale_factor, kernel='cubic'))
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def get_target_top_left(self, size, for_g, idx):
        center = self.target_crop_indices_for_g[idx] if for_g else self.target_crop_indices_for_d[idx]
        row, col = int(center / self.target_in_cols), center % self.target_in_cols
        top, left = min(max(0, row - size // 2), self.target_in_rows - size), min(max(0, col - size // 2), self.target_in_cols - size)
        return top - top % 2, left - left % 2

    def ref_get_top_left(self, size, for_g, img_idx, iter_idx):
        # print(img_idx)
        # print(iter_idx)
        center = self.ref_crop_imgs[img_idx][0][iter_idx] if for_g else self.ref_crop_imgs[img_idx][1][iter_idx]
        in_cols = self.ref_imgs_size[img_idx][1]
        in_rows = self.ref_imgs_size[img_idx][0]

        row, col = int(center / in_cols), center % in_cols
        top, left = min(max(0, row - size // 2), in_rows - size), min(max(0, col - size // 2), in_cols - size)
        return top - top % 2, left - left % 2

    def target_next_crop(self, for_g, idx):
        size = self.g_input_shape if for_g else self.d_input_shape
        scale_cnt = 2
        scale_idx = idx % (scale_cnt * self.conf.batch_size)
        if scale_idx < scale_cnt:
            size = self.g_input_shape//scale_cnt if for_g else self.d_input_shape//scale_cnt
        top, left = self.get_target_top_left(size, for_g, idx)
        crop_im = self.target_image[top:top + size, left:left + size, :]
        return crop_im

    def ref_next_crop(self, img_idx, iter_idx, for_g, idx=0):
        size = self.g_input_shape if for_g else self.d_input_shape
        scale_cnt = 2
        scale_idx = idx % (scale_cnt * self.conf.batch_size)
        if scale_idx < scale_cnt:
            size = self.g_input_shape//scale_cnt if for_g else self.d_input_shape//scale_cnt
        top, left = self.ref_get_top_left(size, for_g, img_idx, iter_idx)
        crop_im = self.ref_imgs[img_idx][top:top + size, left:left + size, :]
        return crop_im

    def __len__(self):
        # return self.conf.train_iters * self.conf.batch_size + self.conf.train_sr_iters * self.conf.batch_size
        return self.conf.train_iters * self.conf.batch_size

    def __getitem__(self, idx):
        if self.conf.ref_continue_mode:
            unit_count = math.ceil(self.conf.train_iters * self.conf.batch_size / self.conf.ref_count * 1.)
            img_idx = idx // unit_count
            iter_idx = idx - img_idx * unit_count
        else:
            img_idx = idx % self.conf.ref_count
            iter_idx = (idx - img_idx) // self.conf.ref_count

        # target
        t_g_in = self.target_next_crop(for_g=True, idx=idx)
        t_d_in = self.target_next_crop(for_g=False, idx=idx)
        t_d_bq_up = imresize(im=t_d_in, scale_factor=self.conf.scale, kernel='cubic')
        # ref
        ref_g_in = self.ref_next_crop(img_idx, iter_idx, for_g=True, idx=idx)
        ref_g_bq_dn = imresize(im=ref_g_in, scale_factor=1.0/self.conf.scale, kernel='cubic')
        ref_g_bq_dn_up = imresize(im=ref_g_bq_dn, scale_factor=self.conf.scale, kernel='cubic')
        ref_d_in = self.ref_next_crop(img_idx, iter_idx, for_g=False, idx=idx)
        ref_d_bq_up = imresize(im=ref_d_in, scale_factor=self.conf.scale, kernel='cubic')

        # return {'tar_big_patch': np2tensor(t_g_in),
        #         'tar_small_patch': np2tensor(t_d_in),
        #         'tar_small_bq_up': np2tensor(t_d_bq_up),
        #         'ref_big_patch': np2tensor(ref_g_in),
        #         'ref_small_patch': np2tensor(ref_d_in),
        #         'ref_small_bq_up': np2tensor(ref_d_bq_up)}

        return {'HR': np2tensor(t_g_in),
                'LR': np2tensor(t_d_in),
                'LR_bicubic': np2tensor(t_d_bq_up),
                'ref_big_patch': np2tensor(ref_g_in),
                'ref_big_bq_dn_up': np2tensor(ref_g_bq_dn_up),
                'ref_small_patch': np2tensor(ref_d_in),
                'ref_small_bq_up': np2tensor(ref_d_bq_up)}
