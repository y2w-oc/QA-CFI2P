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
from config import KittiConfiguration
from utils import ground_segmentation


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]

                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]


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


class KittiDataset(data.Dataset):
    def __init__(self, config, mode):
        super(KittiDataset, self).__init__()
        self.dataset_root = config.dataset_root
        self.data_color = config.data_color
        self.data_velodyne = config.data_velodyne
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
        self.dataset = self.make_kitti_dataset()
        self.calib_helper = KittiCalibHelper(config.dataset_root)

        self.num_node = config.num_node
        self.config = config

        print("%d samples in %s set..." % (len(self.dataset), mode))

    def make_kitti_dataset(self):
        dataset = []
        if self.mode == 'train':
            seq_list = [1]
        elif self.mode == 'val' or self.mode == 'test':
            seq_list = [9]
        else:
            raise Exception('Invalid mode...')

        for seq in seq_list:
            img2_folder = os.path.join(self.dataset_root, self.data_color, 'sequences/', '%02d' % seq, 'image_2')
            img3_folder = os.path.join(self.dataset_root, self.data_color, 'sequences/', '%02d' % seq, 'image_3')
            pc_folder = os.path.join(self.dataset_root, self.data_velodyne, 'sequences/', '%02d' % seq, 'voxel0.1-SNr0.6')

            num = round(len(os.listdir(img2_folder)))
            # select some data as validation dataset
            if self.mode == 'val':
                num = 100

            for i in range(num):
                dataset.append((img2_folder, pc_folder, seq, i, 'P2'))
                dataset.append((img3_folder, pc_folder, seq, i, 'P3'))
        return dataset

    def downsample_pc(self, pc_np, intensity_np, sn_np):
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
        sn_np = sn_np[:, choice_idx]
        return pc_np, intensity_np, sn_np

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
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

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
        return len(self.dataset)

    def __getitem__(self, index):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)

        img_folder, pc_folder, seq, seq_i, key = self.dataset[index]
        img = np.load(os.path.join(img_folder, '%06d.npy' % seq_i))
        data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i))
        # print(data.shape)
        intensity = data[3:4, :]
        sn = data[4:, :]
        pc = data[0:3, :]

        # _, pc = ground_segmentation(pc.T)

        # <------ convert velodyne coordinates to camera coordinates ------>
        P_Tr = np.dot(self.calib_helper.get_matrix(seq, key),
                      self.calib_helper.get_matrix(seq, 'Tr'))
        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]
        sn = np.dot(P_Tr[0:3, 0:3], sn)

        # <------ matrix: camera intrinsics ------>
        K = self.calib_helper.get_matrix(seq, key + '_K')

        # <------ sampling a specified number of points ------>
        pc, intensity, sn = self.downsample_pc(pc, intensity, sn)

        # # <------ crop the useless pixels (the sky) ------>
        # img_crop_dy = 76
        # img = img[img_crop_dy:, :, :]
        # K = self.camera_matrix_cropping(K, dx=0, dy=img_crop_dy)

        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        K = self.camera_matrix_scaling(K, 0.5)

        # <------ cropping a random patch from image ------>
        if self.mode == 'train':
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H, img_crop_dx:img_crop_dx + self.img_W, :]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        # <------ solve the PnP problem at 1/4 scale of the input image ------>
        K = self.camera_matrix_scaling(K, 0.25)

        if self.mode == 'train':
            img = self.augment_img(img)

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
        sn = np.dot(P[0:3, 0:3], sn)

        # <------ sample the firs-level downsampled points, namely node ------>
        node_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice(pc.shape[1],\
                                                  self.num_node * 8, replace=False)], k=self.num_node)

        # <------ construct the node-to-point index ------>
        if self.config.use_gnn_embedding:
            kdtree = cKDTree(pc.T)
            _, I = kdtree.query(pc.T, k=16)
        else:
            kdtree = cKDTree(node_np.T)
            _, I = kdtree.query(pc.T, k=1)

        return {
                'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'intensity': torch.from_numpy(intensity.astype(np.float32)),
                'sn': torch.from_numpy(sn.astype(np.float32)),
                'K': torch.from_numpy(K.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),

                'pc_mask': torch.from_numpy(is_in_picture),
                'point_xy': torch.from_numpy(xy).long(),
                'point_xy_float': torch.from_numpy(pc_[0:2, :]).float(),
                'point_z':torch.from_numpy(pc_[2, :]).float(),

                'pt2node': torch.from_numpy(I).long(),
                'node': torch.from_numpy(node_np).float()
               }


def func_test(dataset, idx_list):
    for i in idx_list:
        dataset[i]


def main_test(dataset):
    thread_num = 12
    idx_list_list = []
    for i in range(thread_num):
        idx_list_list.append([])
    kitti_threads = []
    for i in range(len(dataset)):
        thread_seq_list = [i]
        idx_list_list[int(i % thread_num)].append(i)

    for i in range(thread_num):
        kitti_threads.append(Process(target=func_test,
                                     args=(dataset,
                                           idx_list_list[i])))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


# <------ debug ------>
if __name__ == '__main__':
    config = KittiConfiguration("../../kitti")
    dataset = KittiDataset(config, 'train')
    dataset[0]
    # for i in range(len(dataset)):
    #     print(i)
    #     dataset[i]

    # main_test(dataset)
