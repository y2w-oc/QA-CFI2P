import torch
import torch.nn as nn
import numpy as np
import torch_scatter
import torch.nn.functional as F
import scipy.io as scio
from .pointnet_util import index_points, square_distance


class MiniPointNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(MiniPointNet, self).__init__()

        self.layer_1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm1d(out_channels),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer_2 = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm1d(out_channels),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer_3 = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm1d(out_channels),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))


    def forward(self, x: torch.Tensor):
        """
        Input:
            x: source point
        Output:
            x: features
        """
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


class GroupPointTransformer(nn.Module):
    """
        A modified Pint Transformer layer that is named as Attention Aggregation in our paper
    """
    def __init__(self, d_points=3, d_model=128):
        super(GroupPointTransformer, self).__init__()
        self.fc1_0 = nn.Conv1d(d_points, d_model, kernel_size=1, stride=1, padding=0)
        self.fc1_1 = nn.Conv1d(d_points, d_model, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(d_model, d_points, kernel_size=1, stride=1, padding=0)

        self.fc_delta = nn.Sequential(
            nn.Conv1d(3, d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        )
        self.w_qs = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.w_ks = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.w_vs = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.d_model = d_model

    def forward(self, xyz, xyz_features, node, node_features, idx):
        """
        Input:
            xyx: source point
            xyz_features: point feature
            node: source node
            node_features: node features
            idx: index from xyz -> node (grouping index)
        Output:
            res: the attentional features of different point sets
        """
        b, n, m = xyz.shape[0], xyz.shape[2], node.shape[2]
        pre = node_features  # b x f x m
        x = self.fc1_0(xyz_features)  # b x f x m
        xx = self.fc1_1(node_features)  # b x f x m

        q = self.w_qs(xx) # b,f,m
        k = self.w_ks(x) # b,f,n
        v = self.w_vs(x) # b,f,n

        q = torch.gather(q, index=idx.unsqueeze(1).expand(b, self.d_model, n), dim=2) # b,f,n

        pc_centers = torch.gather(node, index=idx.unsqueeze(1).expand(b, 3, n), dim=2) # b,3,n

        pos_enc = self.fc_delta(xyz - pc_centers)  # b x 3 x n

        attn = self.fc_gamma(q - k + pos_enc)

        # scattered softmax
        f = k.shape[1]
        attn = attn / np.sqrt(f)
        # prevent data overflow
        group_max, _ = torch_scatter.scatter_max(attn, idx.unsqueeze(1).expand(b, f, n), dim=2)
        scattered_max = torch.gather(group_max, index=idx.unsqueeze(1).expand(b, f, n), dim=2)
        attn = (attn - scattered_max).exp()

        att_local_sum = torch_scatter.scatter_sum(attn, idx.unsqueeze(1).expand(b, f, n), dim=2)
        scattered_local_sum = torch.gather(att_local_sum, index=idx.unsqueeze(1).expand(b, f, n), dim=2) # b,f,n
        attn = attn / scattered_local_sum

        res = attn * (v + pos_enc)

        # weighted features
        res = torch_scatter.scatter_sum(res, idx.unsqueeze(1).expand(b, f, n), dim=2)

        res = self.fc2(res) + pre
        return res


class KnnPointTransformer(nn.Module):
    """
        Standard vector attention Point Transformer (ICCV'2021) layer
    """
    def __init__(self, d_points=3, d_model=128, k=16):
        super(KnnPointTransformer, self).__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz, features):
        """
        Input:
            xyx: source point
            features: point feature
        Output:
            res: features with larger perceptive field (KNN)
        """

        # <------change the format to adapt to pointnet_util------>
        xyz = xyz.permute(0,2,1)
        features = features.permute(0,2,1)

        # <------knn------>
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre

        res = res.permute(0, 2, 1)
        return res


class SiameseResMLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_block=5):
        super(SiameseResMLP, self).__init__()

        self.mlps = nn.ModuleList()

        self.mlps.append(ConvBNReLURes1D(in_channels, out_channels))

        for _ in range(num_block-1):
            layer = ConvBNReLURes1D(out_channels, out_channels)
            self.mlps.append(layer)

        layer = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                              nn.BatchNorm1d(out_channels),
                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
                              nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0))

        self.mlps.append(layer)

    def forward(self, x):
        for layer in self.mlps:
            x = layer(x)
        return x


class ConvBNReLURes1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ConvBNReLURes1D, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels)
        )
        self.final_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.final_relu(self.net(x) + self.shortcut(x))

