from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from scipy.signal import convolve2d
from utils.img_utils import read_image, np2tensor, create_gradient_map, create_probability_map, nn_interpolation
from utils.imresize import imresize

import math
import numpy as np
import random


def gen_target_test_dataloader(conf, target_path, tar_gt=''):
    dataset = DownSampleTestDatasets(conf, target_path, tar_gt_path=tar_gt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def gen_test_dataloader(conf, target_path, ref_path, tar_gt='', ref_gt=''):
    dataset = RdSRTestDatasets(conf, target_path, ref_path, tar_gt_path=tar_gt, ref_gt_path=ref_gt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


def gen_test_dataset(conf, target_path, ref_path, tar_gt='', ref_gt=''):
    dataset = RdSRTestDatasets(conf, target_path, ref_path, tar_gt_path=tar_gt, ref_gt_path=ref_gt)
    return dataset


def gen_train_dataloader_adaptive(conf, target_path, ref_path):
    # dataset = RdSRMultiGeneratorS(conf, target_path, ref_path)
    dataset = RdSRDataGenerator(conf, target_path, ref_path)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)

    return dataloader


def gen_downsample_dataloader_adaptive(conf, target_path, target_gt_path):
    dataset = DownSampleDataGenerator(conf, target_path, target_gt_path)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

    return dataloader

def gen_dn_train_dataloader(conf, target_path):
    dataset = RdSRDnDataGenerator(conf, target_path)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)

    return dataloader


class RdSRTestDatasets(Dataset):

    def __init__(self, conf, tar_path, ref_path, tar_gt_path='', ref_gt_path=''):
        np.random.seed(conf.random_seed)
        self.conf = conf
        self.target_image = read_image(tar_path)
        self.target_gt_image = read_image(tar_gt_path)

        self.ref_path = ref_path
        self.ref_imgs = []

        for path in ref_path:
            tmp_ref_img = read_image(path)
            tmp_ref_img = self.shave_edges(conf.scale, tmp_ref_img)
            self.ref_imgs.append(tmp_ref_img)

        self.ref_gt_path = ref_gt_path
        if ref_gt_path:
            self.ref_gt_imgs = []
            for path in ref_gt_path:
                tmp_ref_img = read_image(path)
                tmp_ref_img = self.shave_edges(conf.scale, tmp_ref_img)
                self.ref_gt_imgs.append(tmp_ref_img)

    @staticmethod
    def shave_edges(scale_factor, img):
        sf = scale_factor
        shape = img.shape
        img = img[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else img
        img = img[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else img
        return img

    def __len__(self):
        # return self.conf.ref_count
        return len(self.ref_path)

    def __getitem__(self, idx):
        if self.ref_gt_path:
            data = {
                'Target_Img': np2tensor(self.target_image),
                'Target_Gt': np2tensor(self.target_gt_image),
                'Ref_Img': np2tensor(self.ref_imgs[idx]),
                'Ref_Gt': np2tensor(self.ref_gt_imgs[idx])
            }
        else:
            tar_img = np2tensor(self.target_image)
            tar_up_bq_img = np2tensor(imresize(self.target_image, scale_factor=self.conf.scale))
            tar_gt = np2tensor(self.target_gt_image)
            ref_img = np2tensor(self.ref_imgs[idx])

            data = {
                'Target_Img': tar_img,
                'Target_Gt': tar_gt,
                'Target_Bq_Img': tar_up_bq_img,
                'Ref_Img': ref_img
            }

        return data


class DownSampleTestDatasets(Dataset):

    def __init__(self, conf, tar_path, tar_gt_path=''):
        np.random.seed(conf.random_seed)
        self.conf = conf
        self.target_image = read_image(tar_path)
        self.target_gt_image = read_image(tar_gt_path)

    def __len__(self):
        return self.conf.ref_count

    def __getitem__(self, idx):
        data = {
            'Target_Img': np2tensor(self.target_image),
            'Target_Gt': np2tensor(self.target_gt_image)
        }

        return data


class RdSRDataGenerator(Dataset):

    def __init__(self, conf, tar_path, ref_path):
        np.random.seed(conf.random_seed)
        self.conf = conf

        # time_seed = int(datetime.utcnow().timestamp())
        # print(f'time seed:{time_seed}')
        # np.random.seed(time_seed % 2**32)

        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = int(conf.input_crop_size * (1/conf.scale))

        if tar_path:
            # divide 255 is for create probability map
            self.target_image = read_image(tar_path) / 255.
            # self.target_shave_edges(conf.scale, False)
            self.target_shave_edges(conf.scale, True)
            self.target_in_rows, self.target_in_cols = self.target_image.shape[0:2]
            self.target_crop_indices_for_g = self.make_target_list_of_crop_indices()

            self.target_image = read_image(tar_path)
            # self.target_shave_edges(conf.scale, False)
            self.target_shave_edges(conf.scale, True)

        self.ref_path = ref_path
        self.ref_imgs = []
        self.ref_imgs_size = []
        self.ref_crop_imgs = []
        if ref_path:
            for ref_img_path in ref_path:
                # divide 255 is for create probability map
                tmp_ref_img = read_image(ref_img_path) / 255.
                # tmp_ref_img = self.ref_shave_edges(conf.scale, False, tmp_ref_img)
                tmp_ref_img = self.ref_shave_edges(conf.scale, True, tmp_ref_img)
                tmp_in_rows, tmp_in_cols = tmp_ref_img.shape[0:2]
                g_crop_indices, d_crop_indices = self.make_list_of_crop_indices_m(tmp_ref_img)
                self.ref_crop_imgs.append((g_crop_indices, d_crop_indices))
                self.ref_imgs_size.append((tmp_in_rows, tmp_in_cols))

                tmp_ref_img = read_image(ref_img_path)
                # tmp_ref_img = self.ref_shave_edges(conf.scale, False, tmp_ref_img)
                tmp_ref_img = self.ref_shave_edges(conf.scale, True, tmp_ref_img)
                self.ref_imgs.append(tmp_ref_img)

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
        # prob_map_big, prob_map_sml = self.create_target_prob_maps(1/self.conf.scale)
        prob_map_big = self.create_target_prob_maps(1/self.conf.scale)

        # crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        # crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        crop_indices_for_g = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g

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
        # loss_map_sml = create_gradient_map(imresize(im=self.target_image, scale_factor=scale_factor, kernel='cubic'))
        # prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        # prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big

    def get_top_left(self, size, for_g, img_idx, iter_idx):
        # print(img_idx)
        # print(iter_idx)
        center = self.ref_crop_imgs[img_idx][0][iter_idx] if for_g else self.ref_crop_imgs[img_idx][1][iter_idx]
        in_cols = self.ref_imgs_size[img_idx][1]
        in_rows = self.ref_imgs_size[img_idx][0]

        row, col = int(center / in_cols), center % in_cols
        top, left = min(max(0, row - size // 2), in_rows - size), min(max(0, col - size // 2), in_cols - size)
        return top - top % 2, left - left % 2

    def get_target_top_left(self, size, for_g, idx):
        # center = self.target_crop_indices_for_g[idx] if for_g else self.target_crop_indices_for_d[idx]
        center = self.target_crop_indices_for_g[idx]
        row, col = int(center / self.target_in_cols), center % self.target_in_cols
        top, left = min(max(0, row - size // 2), self.target_in_rows - size), min(max(0, col - size // 2), self.target_in_cols - size)
        return top - top % 2, left - left % 2

    def next_crop(self, img_idx, iter_idx, for_g):
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, img_idx, iter_idx)
        crop_im = self.ref_imgs[img_idx][top:top + size, left:left + size, :]
        return crop_im

    def target_next_crop(self, for_g, idx):
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_target_top_left(size, for_g, idx)
        crop_im = self.target_image[top:top + size, left:left + size, :]
        return crop_im

    @staticmethod
    def get_patch(img, patch_size=64, scale=1):
        th, tw = img.shape[:2]  ## HR image

        tp = round(scale * patch_size)

        tx = random.randrange(0, (tw - tp))
        ty = random.randrange(0, (th - tp))

        return img[ty:ty + tp, tx:tx + tp, :]

    def __len__(self):
        # return self.conf.train_iters * self.conf.batch_size + self.conf.train_sr_iters * self.conf.batch_size
        return self.conf.train_iters * self.conf.batch_size

    def __getitem__(self, idx):
        if self.conf.ref_continue_mode:
            unit_count = math.ceil(self.conf.train_iters * self.conf.batch_size / self.conf.ref_count * 1.)
            img_idx = idx // unit_count
            iter_idx = idx - img_idx * unit_count
        else:
            # for ref images mix into iterations
            img_idx = idx % self.conf.ref_count
            iter_idx = (idx - img_idx) // self.conf.ref_count

        ref_in = self.next_crop(img_idx, iter_idx, for_g=True)
        ref_bq_dn = imresize(im=ref_in, scale_factor=1/self.conf.scale, kernel='cubic')
        ref_in = np2tensor(ref_in)
        ref_bq_dn = np2tensor(ref_bq_dn)

        t_d_in = self.target_next_crop(for_g=False, idx=idx)
        t_d_bq = imresize(im=t_d_in, scale_factor=self.conf.scale, kernel='cubic')
        t_d_in = np2tensor(t_d_in)
        t_d_bq = np2tensor(t_d_bq)

        return {'Ref_HR': ref_in,
                'Ref_bq_dn': ref_bq_dn,
                'Tar_LR': t_d_in,
                'Tar_bq_up': t_d_bq}


class RdSRDnDataGenerator(Dataset):

    def __init__(self, conf, tar_path):
        np.random.seed(conf.random_seed)
        self.conf = conf

        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = int(conf.input_crop_size * (1/conf.scale))

        if tar_path:
            self.target_image = read_image(tar_path) / 255.
            self.target_shave_edges(conf.scale, True)
            self.target_in_rows, self.target_in_cols = self.target_image.shape[0:2]
            self.target_crop_indices_for_g = self.make_target_list_of_crop_indices()

            self.target_image = read_image(tar_path)
            self.target_shave_edges(conf.scale, True)

    def target_shave_edges(self, scale, real_image):
        if not real_image:
            self.target_image = self.target_image[10:-10, 10:-10, :]
        shape = self.target_image.shape
        self.target_image = self.target_image[:-(shape[0] % scale), :, :] if shape[0] % scale > 0 else self.target_image
        self.target_image = self.target_image[:, :-(shape[1] % scale), :] if shape[1] % scale > 0 else self.target_image

    def make_target_list_of_crop_indices(self):
        iterations = self.conf.train_iters * self.conf.batch_size
        prob_map_big = self.create_target_prob_maps(1/self.conf.scale)

        crop_indices_for_g = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g

    def create_target_prob_maps(self, scale_factor):
        loss_map_big = create_gradient_map(self.target_image)
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)

        return prob_map_big

    def get_top_left(self, size, for_g, img_idx, iter_idx):
        # print(img_idx)
        # print(iter_idx)
        center = self.ref_crop_imgs[img_idx][0][iter_idx] if for_g else self.ref_crop_imgs[img_idx][1][iter_idx]
        in_cols = self.ref_imgs_size[img_idx][1]
        in_rows = self.ref_imgs_size[img_idx][0]

        row, col = int(center / in_cols), center % in_cols
        top, left = min(max(0, row - size // 2), in_rows - size), min(max(0, col - size // 2), in_cols - size)
        return top - top % 2, left - left % 2

    def get_target_top_left(self, size, for_g, idx):
        center = self.target_crop_indices_for_g[idx]
        row, col = int(center / self.target_in_cols), center % self.target_in_cols
        top, left = min(max(0, row - size // 2), self.target_in_rows - size), min(max(0, col - size // 2), self.target_in_cols - size)
        return top - top % 2, left - left % 2

    def target_next_crop(self, for_g, idx):
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_target_top_left(size, for_g, idx)
        crop_im = self.target_image[top:top + size, left:left + size, :]
        return crop_im

    @staticmethod
    def get_patch(img, patch_size=64, scale=1):
        th, tw = img.shape[:2]

        tp = round(scale * patch_size)

        tx = random.randrange(0, (tw - tp))
        ty = random.randrange(0, (th - tp))

        return img[ty:ty + tp, tx:tx + tp, :]

    def __len__(self):
        return self.conf.train_iters * self.conf.batch_size

    def __getitem__(self, idx):
        # TODO: this is for train one image for at one period
        t_d_in = self.target_next_crop(for_g=False, idx=idx)
        t_d_bq = imresize(im=t_d_in, scale_factor=self.conf.scale, kernel='cubic')
        t_d_in = np2tensor(t_d_in)
        t_d_bq = np2tensor(t_d_bq)

        return {'Tar_LR': t_d_in,
                'Tar_bq_up': t_d_bq}


class DownSampleDataGenerator(Dataset):

    def __init__(self, conf, tar_path, gt_path):
        np.random.seed(conf.random_seed)
        self.conf = conf

        # time_seed = int(datetime.utcnow().timestamp())
        # print(f'time seed:{time_seed}')
        # np.random.seed(time_seed % 2**32)

        self.input_shape = conf.input_crop_size

        if tar_path:
            self.target_image = read_image(tar_path) / 255.
            self.target_shave_edges(conf.scale, True)
            self.target_in_rows, self.target_in_cols = self.target_image.shape[0:2]
            self.target_crop_indices = self.make_target_list_of_crop_indices()

            self.target_image = read_image(tar_path)
            self.target_shave_edges(conf.scale, True)

        if gt_path:
            self.gt_image = read_image(gt_path)

    def target_shave_edges(self, scale, real_image):
        if not real_image:
            self.target_image = self.target_image[10:-10, 10:-10, :]
        shape = self.target_image.shape
        self.target_image = self.target_image[:-(shape[0] % scale), :, :] if shape[0] % scale > 0 else self.target_image
        self.target_image = self.target_image[:, :-(shape[1] % scale), :] if shape[1] % scale > 0 else self.target_image

    def make_target_list_of_crop_indices(self):
        iterations = self.conf.train_iters * self.conf.batch_size
        prob_map = self.create_target_prob_maps()

        crop_indices = np.random.choice(a=len(prob_map), size=iterations, p=prob_map)
        return crop_indices

    def create_target_prob_maps(self):
        loss_map = create_gradient_map(self.target_image)
        prob_map = create_probability_map(loss_map, self.input_shape)
        return prob_map

    def get_top_left(self, size, for_g, img_idx, iter_idx):
        center = self.ref_crop_imgs[img_idx][0][iter_idx] if for_g else self.ref_crop_imgs[img_idx][1][iter_idx]
        in_cols = self.ref_imgs_size[img_idx][1]
        in_rows = self.ref_imgs_size[img_idx][0]

        row, col = int(center / in_cols), center % in_cols
        top, left = min(max(0, row - size // 2), in_rows - size), min(max(0, col - size // 2), in_cols - size)
        return top - top % 2, left - left % 2

    def get_target_top_left(self, size, idx):
        center = self.target_crop_indices[idx]
        row, col = int(center / self.target_in_cols), center % self.target_in_cols
        top, left = min(max(0, row - size // 2), self.target_in_rows - size), min(max(0, col - size // 2), self.target_in_cols - size)
        return top - top % 2, left - left % 2

    def target_next_crop(self, idx):
        size = self.input_shape
        top, left = self.get_target_top_left(size, idx)
        crop_im = self.target_image[top:top + size, left:left + size, :]
        top_gt = top * self.conf.scale
        left_gt = left * self.conf.scale
        size_gt = size * self.conf.scale
        crop_gt_im = self.gt_image[top_gt:top_gt + size_gt, left_gt:left_gt + size_gt, :]
        return crop_im, crop_gt_im

    @staticmethod
    def get_patch(img, patch_size=64, scale=1):
        th, tw = img.shape[:2]  ## HR image

        tp = round(scale * patch_size)

        tx = random.randrange(0, (tw - tp))
        ty = random.randrange(0, (th - tp))

        return img[ty:ty + tp, tx:tx + tp, :]

    def __len__(self):
        # return self.conf.train_iters * self.conf.batch_size + self.conf.train_sr_iters * self.conf.batch_size
        return self.conf.train_iters * self.conf.batch_size

    def __getitem__(self, idx):
        t_in, t_gt_in = self.target_next_crop(idx=idx)
        t_d_bq = imresize(im=t_in, scale_factor=self.conf.scale, kernel='cubic')
        t_in = np2tensor(t_in)
        t_gt_in = np2tensor(t_gt_in)
        t_d_bq = np2tensor(t_d_bq)

        return {'Tar_LR': t_in,
                'Tar_bq_up': t_d_bq,
                'Tar_Gt': t_gt_in}

