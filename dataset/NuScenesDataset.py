import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from multiprocessing import Process
import open3d
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import torch_scatter
import time
import sys
from scipy.spatial import cKDTree
sys.path.append("..")
from config import NuScenesConfiguration


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int64)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i + 1], pts))
        return farthest_pts, farthest_pts_idx


class NuScenesDataset(data.Dataset):
    def __init__(self, config, mode):
        super(NuScenesDataset, self).__init__()
        self.dataset_root = config.dataset_root
        self.mode = mode
        self.num_pt = config.num_pt
        self.img_H = config.cropped_img_H
        self.img_W = config.cropped_img_W
        self.patch_size = config.patch_size
        self.P_Tx_amplitude = config.P_Tx_amplitude
        self.P_Ty_amplitude = config.P_Ty_amplitude
        self.P_Tz_amplitude = config.P_Tz_amplitude
        self.P_Rx_amplitude = config.P_Rx_amplitude
        self.P_Ry_amplitude = config.P_Ry_amplitude
        self.P_Rz_amplitude = config.P_Rz_amplitude

        self.farthest_sampler = FarthestSampler(dim=3)

        if self.mode == 'train':
            self.pc_path = os.path.join(self.dataset_root, 'train', 'PC')
            self.img_path = os.path.join(self.dataset_root, 'train', 'img')
            self.K_path = os.path.join(self.dataset_root, 'train', 'K')
        elif self.mode == "test" or self.mode == "val":
            self.pc_path = os.path.join(self.dataset_root, 'test', 'PC')
            self.img_path = os.path.join(self.dataset_root, 'test', 'img')
            self.K_path = os.path.join(self.dataset_root, 'test', 'K')
        else:
            assert False, "Mode error! Mode should be 'train', 'test' or 'val'"

        self.length = len(os.listdir(self.pc_path))

        if self.mode == "val":
            self.length = 400

        self.num_node = config.num_node
        self.config = config

        print("%d samples in %s set..." % (self.length, mode))

    def __len__(self):
        return self.length

    def downsample_pc(self, pc_np, intensity_np):
        if pc_np.shape[1] >= self.num_pt:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pt, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pt:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pt - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        return pc_np, intensity_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(np.uint8(img_np))))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def generate_random_transform(self):
        """
        Generate a random transform matrix according to the configuration
        """
        t = [random.uniform(-self.P_Tx_amplitude, self.P_Tx_amplitude),
             random.uniform(-self.P_Ty_amplitude, self.P_Ty_amplitude),
             random.uniform(-self.P_Tz_amplitude, self.P_Tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t
        return P_random

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)

        data = np.load(os.path.join(self.pc_path, '%06d.npy' % index))
        img = np.load(os.path.join(self.img_path, '%06d.npy' % index))
        K = np.load(os.path.join(self.K_path, '%06d.npy' % index))

        pc = data[0:3, :]
        intensity = data[3:, :]
        # <------ sampling a specified number of points ------>
        pc, intensity = self.downsample_pc(pc, intensity)

        # # <------ cropping a random patch from image ------>
        # if self.mode == 'train':
        #     img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
        #     img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        # else:
        #     img_crop_dx = int((img.shape[1] - self.img_W) / 2)
        #     img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        # img = img[img_crop_dy:img_crop_dy + self.img_H, img_crop_dx:img_crop_dx + self.img_W, :]
        # K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        # <------ solve the PnP problem at 1/4 scale of the input image ------>
        K = self.camera_matrix_scaling(K, 0.25)

        if self.mode == 'train':
            img = self.augment_img(img)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # key = cv2.waitKey(0)

        pc_ = np.dot(K, pc)
        pc_mask = np.zeros((1, pc.shape[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.round(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) & (xy[1, :] >= 0) & \
                        (xy[1, :] <= (self.img_H*0.25-1)) & (pc_[2, :] > 0)

        pc_mask[:, is_in_picture] = 1.

        # <------transform the point cloud------>
        P = self.generate_random_transform()
        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        # <------sample some node to extract features------>
        node_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice(pc.shape[1],\
                                                  self.num_node * 8, replace=False)], k=self.num_node)

        if self.config.use_gnn_embedding:
            kdtree = cKDTree(pc.T)
            _, I = kdtree.query(pc.T, k=16)
        else:
            kdtree = cKDTree(node_np.T)
            _, I = kdtree.query(pc.T, k=1)

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'intensity': torch.from_numpy(intensity.astype(np.float32)),
                'K': torch.from_numpy(K.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),

                'pc_mask': torch.from_numpy(is_in_picture),
                'point_xy': torch.from_numpy(xy).long(),
                'point_xy_float': torch.from_numpy(pc_[0:2, :]).float(),

                'pt2node': torch.from_numpy(I).long(),
                'node': torch.from_numpy(node_np).float()}


# <------ debug ------>
if __name__ == '__main__':
    config = NuScenesConfiguration("../../nuscenes2")
    dataset = NuScenesDataset(config, 'train')
    # dataset[0]
    for i in range(100):
        dataset[i]
