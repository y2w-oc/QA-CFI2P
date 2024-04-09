import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
import numpy as np
import scipy.io as scio
from scipy.spatial.transform import Rotation
from config import KittiConfiguration, NuScenesConfiguration
from dataset import KittiDataset, NuScenesDataset
from models import FineI2P
import cv2
import argparse


# <--------- relative rotation and translation error -------->
def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])
    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)))

    return t_diff, angles_diff


if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Image to point Registration')
    parser.add_argument('--dataset', type=str, default='kitti', help=" 'kitti' or 'nuscenes' ")
    args = parser.parse_args()

    # <------Configuration parameters------>
    if args.dataset == "kitti":
        config = KittiConfiguration()
        test_dataset = KittiDataset(config, mode='test')
    elif args.dataset == "nuscenes":
        config = NuScenesConfiguration()
        test_dataset = NuScenesDataset(config, mode='test')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                             drop_last=False, num_workers=10)

    model = FineI2P(config, load_coarse_model=False)
    model = model.cuda()

    # "kitti.pth" is a pretrained model on KITTI Odometry. You can modify this to use other models.
    sate_dict = torch.load("checkpoint/kitti.pth")
    model.load_state_dict(sate_dict)
    model.eval()

    t_diff_set = []
    angles_diff_set = []
    is_success_set = []
    T_pred_set = []
    T_gt_set = []

    t_diff_set_0 = []
    angles_diff_set_0 = []

    t_diff_set_1 = []
    angles_diff_set_1 = []

    with torch.no_grad():
        for step,data in enumerate(tqdm(test_loader)):
            model(data)
            scores = data['fine_scores'][:, :-1, :-1]
            coarse_scores = data['sampled_coarse_scores']
            coarse_scores, top_idx = torch.topk(coarse_scores, k=config.topk_proxy, dim=1)

            pixel_xy = data['sampled_pixel_xy']
            point_xy = data['sampled_point_xy']
            pixel_mask = data['sample_pixel_mask']
            pt_mask = data['sample_pt_mask']

            # <-------------------------- confidence sorting strategy ------------------------>
            corr_per_set = pt_mask.sum(1) * coarse_scores.sum(1)
            corr_per_set = torch.round(corr_per_set).long()

            scores = scores * pixel_mask.unsqueeze(2)
            scores = scores * pt_mask.unsqueeze(1)

            out_scores = data['fine_scores'][:, -1, :-1]

            scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)
            e_xy = scores.unsqueeze(1) * pixel_xy.unsqueeze(3)
            e_xy = e_xy.sum(dim=2)
            point_xy = point_xy * pt_mask.unsqueeze(1)

            sorted_max_scores, sortd_idx = torch.sort(out_scores, dim=1, descending=False)
            b_idx = torch.arange(corr_per_set.shape[0], dtype=torch.long).cuda()
            mm = corr_per_set > 1
            corr_per_set[mm] = corr_per_set[mm] - 1
            scores_thres = sorted_max_scores[b_idx, corr_per_set]
            scores_thres = scores_thres.unsqueeze(1).repeat(1, config.pt_sample_num)
            cardi_mask = out_scores <= scores_thres

            pt_mask = pt_mask * cardi_mask

            # <---------------------------- image-to-point cloud registration -------------------------->
            sampled_point_3d = data['sampled_point_3d']
            patch_td, point_idx = torch.where(pt_mask)
            point_3d = sampled_point_3d[patch_td, :, point_idx]
            point_2d = e_xy[patch_td, :, point_idx]

            # ground truth 2D projected points
            point_2d_gt = point_xy[patch_td, :, point_idx]
            point_2d_gt = point_2d_gt.cpu().numpy()

            # registration
            point_3d = point_3d.cpu().numpy()
            point_2d = point_2d.cpu().numpy()
            K = data['K'].numpy()[0,:,:]
            P = data['P'].numpy()[0, :, :]
            try:
                is_success, R, t, inliers = cv2.solvePnPRansac(point_3d, point_2d, K,
                                                               useExtrinsicGuess=False,
                                                               iterationsCount=500,
                                                               reprojectionError=1,
                                                               flags=cv2.SOLVEPNP_EPNP,  # cv2.SOLVEPNP_P3P  cv2.SOLVEPNP_ITERATIVE
                                                               distCoeffs=None)
                inliers = inliers[:,0]
                inlier_pt = point_2d[inliers,:]
                inlier_pt_gt = point_2d_gt[inliers,:]
            except:
                print("Solve PnP Error!")
                assert False

            R, _ = cv2.Rodrigues(R)
            T_pred = np.eye(4)
            T_pred[0:3, 0:3] = R
            T_pred[0:3, 3:] = t

            # calculate RTE and RRE
            t_diff, angles_diff = get_P_diff(T_pred, P)

            t_diff_set.append(t_diff)
            angles_diff_set.append(angles_diff)
            is_success_set.append(is_success)
            T_pred_set.append(T_pred)
            T_gt_set.append(P)

            if is_success:
                print('RTE: ', t_diff, ' RRE: ', angles_diff)
            else:
                print("PnP solve failed!")
                time.sleep(10)

        # calculate mean RTE and RRE
        t_diff_set = np.array(t_diff_set)
        angles_diff_set = np.array(angles_diff_set)
        scio.savemat("diff_EPnP.mat", {"T_diff": t_diff_set, "R_diff": angles_diff_set})
        successful_mask = (t_diff_set < 5) & (angles_diff_set < 10)
        t_diff_set = t_diff_set[successful_mask]
        angles_diff_set = angles_diff_set[successful_mask]
        print('RTE Mean: ', t_diff_set.mean(), ' RTE Std: ', t_diff_set.std())
        print('RRE Mean: ', angles_diff_set.mean(), ' RRE Std: ', angles_diff_set.std())
