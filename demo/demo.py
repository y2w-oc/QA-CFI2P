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
from config import KittiConfiguration
from dataset import KittiDataset
from models import FineI2P
import cv2
import argparse
from scipy.sparse import coo_matrix
import matplotlib as mpl
import matplotlib.cm as cm
import math


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

    sate_dict = torch.load("../checkpoint/kitti.pth")
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

            point_2d_gt = point_xy[patch_td, :, point_idx]
            point_2d_gt = point_2d_gt.cpu().numpy()

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

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                is_in_picture = data['pc_mask'].squeeze().cpu().numpy()
                xy = data['point_xy_float'].squeeze().cpu().numpy()
                point_z = data['point_z'].squeeze().cpu().numpy()
                img = data['img'].squeeze().cpu().numpy()
                xy2 = np.round(xy[:, is_in_picture] * 4)
                mmm = xy2 < 0
                xy2[mmm] = 0
                img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])),
                                      shape=(160, 512)).toarray()
                img_mask = np.array(img_mask)
                img_mask[img_mask > 0] = 1.

                color_depth = point_z[is_in_picture]
                depth_mask = color_depth > (color_depth.min() + 50)
                color_depth[depth_mask] = (color_depth.min() + 50)
                color_depth = -1 * color_depth
                norm = mpl.colors.Normalize(vmin=color_depth.min(), vmax=color_depth.max())
                cmap = cm.jet
                cm_sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                p_colors = cm_sm.to_rgba(color_depth)[:, 0:3]

                img_p_colors = np.ones((160, 512, 3))
                for i in range(int(is_in_picture.sum())):
                    img_p_colors[int(xy2[1, i]), int(xy2[0, i])] = p_colors[i, :]

                img = img.transpose(1,2,0)
                img = img[:, :, ::-1]

                cv2.namedWindow('img_p', cv2.WINDOW_NORMAL)
                cv2.imshow('img_p', img_p_colors)

                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', img)

                fused_img = np.ones((160*2+10, 512, 3))
                fused_img[0:160,:, :] = img_p_colors
                fused_img[170:, :, :] = img

                threshold = 3

                counter = 0
                for i in range(inlier_pt.shape[0]):
                    y0, x0 = inlier_pt[i,:]
                    y1, x1 = inlier_pt_gt[i, :]
                    dist = math.sqrt((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1))
                    x0 = x0 * 4
                    x1 = x1 * 4
                    y0 = y0 * 4
                    y1 = y1 * 4
                    if dist <= threshold:
                        cv2.line(fused_img,
                                 (int(y0), int(x0)),
                                 (int(y1), int(x1)+170),
                                 (0, 1, 0),
                                 1)
                        counter += 1
                    else:
                        cv2.line(fused_img,
                                 (int(y0), int(x0)),
                                 (int(y1), int(x1) + 170),
                                 (0, 0, 1),
                                 1)
                print("inliers:%d/%d %.4f" % (counter, inlier_pt.shape[0], counter / inlier_pt.shape[0]))
                cv2.namedWindow('fused_img', cv2.WINDOW_NORMAL)
                cv2.imshow('fused_img', fused_img)

            except:
                print("Solve PnP Error!")
                assert False
            R, _ = cv2.Rodrigues(R)

            T_pred = np.eye(4)
            T_pred[0:3, 0:3] = R
            T_pred[0:3, 3:] = t

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

            key = cv2.waitKey(0)


