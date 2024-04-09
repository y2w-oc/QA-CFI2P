import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import cv2
import scipy.io as scio
import torch_scatter
from .CoarseI2P import CoarseI2P
from .PointNN import ConvBNReLURes1D
from .ImageResNet import ResidualBlock
from .PointViT import PointTransformer
sys.path.append("..")
from utils import PositionEncodingSine2D, log_optimal_transport
from .LinearAttention import LinearAttention


class MaskAttention(nn.Module):
    def __init__(self, config):
        super(MaskAttention, self).__init__()
        self.num_attention_heads = config.num_head
        self.attention_head_size = int(config.embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.embed_dim, self.all_head_size)
        self.key = nn.Linear(config.embed_dim, self.all_head_size)
        self.value = nn.Linear(config.embed_dim, self.all_head_size)

        self.out = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj_dropout = nn.Dropout(config.attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_hidden_states, y_hidden_states, mask_x, mask_y):
        mixed_query_layer = self.query(x_hidden_states)
        mixed_key_layer = self.key(y_hidden_states)
        mixed_value_layer = self.value(y_hidden_states)
        mixed_query_layer = mixed_query_layer * mask_x.unsqueeze(2)
        mixed_key_layer = mixed_key_layer * mask_y.unsqueeze(2)
        mixed_value_layer = mixed_value_layer * mask_y.unsqueeze(2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(config.mlp_dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = MaskAttention(config)

    def forward(self, x, y, mask_x=None, mask_y=None):
        if mask_x is None:
            mask_x = torch.ones(x.shape[0:2]).to(x.device) > 0
            mask_y = torch.ones(y.shape[0:2]).to(y.device) > 0

        h = x
        x = self.attention_norm(x)
        y = self.attention_norm(y)
        x = self.attn(x, y, mask_x, mask_y)
        x = h + x

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + x
        return x


class FineI2P(nn.Module):
    """
        Fine part of CFI2P
    """
    def __init__(self, config, load_coarse_model=True):
        super(FineI2P, self).__init__()
        self.config = config
        self.coarse_model = CoarseI2P(config)

        # <------ load a pretrained coarse model ------>
        if load_coarse_model:
            sate_dict = torch.load(config.used_coarse_checkpoint)
            self.coarse_model.load_state_dict(sate_dict)
            self.coarse_model.eval()
        else:
            pass

        self.H_proxy = self.config.image_H // self.config.patch_size
        self.W_proxy = self.config.image_W // self.config.patch_size
        self.img_proxy_num = self.H_proxy * self.W_proxy
        self.pt_sample_num = config.pt_sample_num
        f = config.embed_dim

        # <------ fine features extraction   ------>
        # <------ we add linear-attention between nodes (the first-level downsampled point) and pixels ------>
        self.upsample_img_proxy = torch.nn.Upsample(scale_factor=self.config.patch_size, mode='nearest')
        self.node_fuse_convs = nn.ModuleList()
        self.node_fuse_convs.append(ConvBNReLURes1D(2 * f, f))
        for _ in range(config.node_fuse_res_num - 1):
            self.node_fuse_convs.append(ConvBNReLURes1D(f, f))

        self.node_self_LA = nn.ModuleList()
        self.pixel_to_node_LA = nn.ModuleList()
        self.node_to_pixel_LA = nn.ModuleList()
        self.pixel_self_LA = nn.ModuleList()
        for i in range(config.linear_attention_num):
            self.node_self_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.pixel_to_node_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.node_to_pixel_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.pixel_self_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))

        self.point_fuse_convs = nn.ModuleList()
        self.point_fuse_convs.append(ConvBNReLURes1D(2 * f, f))
        for _ in range(config.pt_head_res_num - 1):
            self.point_fuse_convs.append(ConvBNReLURes1D(f, f))

        self.img_fuse_convs = nn.ModuleList()
        self.img_fuse_convs.append(ResidualBlock(2 * f, f))
        for _ in range(config.img_fuse_res_num - 1):
            self.img_fuse_convs.append(ResidualBlock(f, f))

        self.pixel_pos_embed_convs = nn.ModuleList()
        self.pixel_pos_embed_convs.append(ResidualBlock(f, f))
        for _ in range(config.pixel_pos_embed_convs_res_num - 1):
            self.pixel_pos_embed_convs.append(ResidualBlock(f, f))

        # <------ fine full-attention head ------>
        self.i2p_ca_layers = nn.ModuleList()
        self.p2i_ca_layers = nn.ModuleList()
        self.pt_sa_layers = nn.ModuleList()
        self.img_sa_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_fine):
            self.i2p_ca_layers.append(Block(config))
            self.p2i_ca_layers.append(Block(config))
            self.pt_sa_layers.append(Block(config))
            self.img_sa_layers.append(Block(config))

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # <------ img pixel positional embedding ------>
        self.pixel_pos_encoding = PositionEncodingSine2D(f, (config.image_H, config.image_W))

    def optimal_transport(self, scores):
        return log_optimal_transport(scores, None, None, self.bin_score, iters=self.config.sinkhorn_iters)

    def index_feats(self, feats, idx):
        """
        Input:
            feats: input features data, [B, C, N]
            idx: sample index data, [B, M]
        Output:
            sampled_feats:, indexed feats data, [B, C, M]
        """
        device = feats.device
        b, c, n = feats.size()
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(b, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        sampled_feats = feats[batch_indices, :, idx]
        return sampled_feats

    def calculate_pixel_idx(self, top_idx):
        """
        Input:
            top_idx: the selected pixel proxies
        Output:
            patch_idx: the pixel index (w^2 pixels of each pixel patch)
        """
        with torch.no_grad():
            device = top_idx.device
            b, n, pt_proxy_num = top_idx.shape
            top_idx = top_idx.view(b,-1)
            p = self.config.patch_size
            top_img_proxy_x = top_idx % self.W_proxy
            top_img_proxy_y = top_idx // self.W_proxy
            # top_img_proxy_yx = torch.cat([top_img_proxy_y.unsqueeze(1), top_img_proxy_x.unsqueeze(1)], dim=1)
            bias = torch.linspace(0, p - 1, p).to(device)
            patch_x = (top_img_proxy_x.unsqueeze(-1) * p + \
                       bias.unsqueeze(0).unsqueeze(0)).unsqueeze(-2).unsqueeze(-1)
            patch_y = (top_img_proxy_y.unsqueeze(-1) * p + \
                       bias.unsqueeze(0).unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            # patch_pt_yx = torch.cat([patch_y.repeat(1, 1, 1, p, 1), patch_x.repeat(1, 1, p, 1, 1)], dim=-1)
            patch_idx = patch_y * self.config.image_W + patch_x
            patch_idx = patch_idx.view(b, n, pt_proxy_num, -1)
            patch_idx = patch_idx.permute(0,2,1,3).long().contiguous()
            patch_idx = patch_idx.view(b,pt_proxy_num,-1)

        return patch_idx

    def fine_matching_loss(self, fine_scores, fine_gt):
        """
        Calculating loss for fine point-pixel matching
        :param fine_scores: Predicted matching scores, [B,N+1,M+1]
        :param fine_gt: The ground truth, [B,N+1,M+1]
        :return: Calculated loss
        """
        y = fine_gt * fine_scores
        loss = torch.sum(-y, dim=(1, 2)) / torch.sum(fine_gt, dim=(1, 2))
        loss = loss.sum() / fine_scores.shape[0]
        return loss

    def forward(self, data_batch):
        # fixed the parameters of the coarse model
        with torch.no_grad():
            self.coarse_model(data_batch)

        img_feat = data_batch['img_feat']
        pt_feat = data_batch['pt_feat']
        node_feat = data_batch['node_feat']

        device = pt_feat.device

        img_proxy = data_batch['img_proxy']
        img_proxy = img_proxy.permute(0, 2, 1)
        pt_proxy = data_batch['pt_proxy']
        pt_proxy = pt_proxy.permute(0, 2, 1)
        scores = data_batch['scores'][:,:-1,:-1]
        scores_gt = data_batch['scores_gt'][:,:-1,:-1]

        # <------ create the index from source point to point proxy ------>
        node2proxy = data_batch['node2proxy']
        pt2node = data_batch['pt2node'].to(device)
        pt2ptproxy = torch.gather(node2proxy, index=pt2node, dim=1)

        # <------ fuse the point proxy with node features via grouping ------>
        f = pt_proxy.shape[1]
        b, n = node2proxy.shape[0], node2proxy.shape[1]
        scattered_node_proxy_feat = torch.gather(pt_proxy, index=node2proxy.unsqueeze(1).expand(b, f, n), dim=2)
        fused_node_feat = torch.cat([node_feat, scattered_node_proxy_feat], dim=1)  # [b, 2f, 1280]

        for layer in self.node_fuse_convs:
            fused_node_feat = layer(fused_node_feat)

        # <------fuse the pixel proxy with pixel features via nearest upsampling------>
        f = img_proxy.shape[1]
        img_proxy_4d = img_proxy.reshape(b, f, self.H_proxy, self.W_proxy)
        scattered_img_proxy_feat = self.upsample_img_proxy(img_proxy_4d)
        # scattered_img_proxy_feat = scattered_img_proxy_feat + self.img_position_embeddings
        fused_img_feat = torch.cat([img_feat, scattered_img_proxy_feat], dim=1)

        for layer in self.img_fuse_convs:
            fused_img_feat = layer(fused_img_feat)

        fused_img_feat = fused_img_feat.view(b, f, -1)
        fused_img_feat = fused_img_feat.permute(0, 2, 1)
        fused_node_feat = fused_node_feat.permute(0, 2, 1)

        # <------ linear attention between fused_node_feat and fused_img_feat(pixel-level) ------>
        for i in range(self.config.linear_attention_num):
            layer = self.pixel_to_node_LA[i]
            fused_node_feat = layer(fused_node_feat, fused_img_feat)
            layer = self.node_to_pixel_LA[i]
            fused_img_feat = layer(fused_img_feat, fused_node_feat)
            layer = self.node_self_LA[i]
            fused_node_feat = layer(fused_node_feat, fused_node_feat)
            layer = self.pixel_self_LA[i]
            fused_img_feat = layer(fused_img_feat, fused_img_feat)

        fused_img_feat = fused_img_feat.permute(0, 2, 1)
        fused_img_feat = fused_img_feat.view(b, f, self.config.image_H, self.config.image_W)
        fused_node_feat = fused_node_feat.permute(0, 2, 1)

        # <------ pixel-level feature of the image (1/4 size) ------>
        fused_img_feat = self.pixel_pos_encoding(fused_img_feat)
        for layer in self.pixel_pos_embed_convs:
            fused_img_feat = layer(fused_img_feat)

        # <------ fuse the node features with point features via grouping ------>
        f = fused_node_feat.shape[1]
        b, n = pt2node.shape[0], pt2node.shape[1]
        scattered_pt_node_feat = torch.gather(fused_node_feat, index=pt2node.unsqueeze(1).expand(b, f, n), dim=2)
        fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)

        for layer in self.point_fuse_convs:
            fused_pt_feat = layer(fused_pt_feat)

        # <====== sample n points in each point set (proxy) ======>
        # <------ repeat the points to fill data batches ------>
        with torch.no_grad():
            if self.training:
                pos_pt_proxy = scores_gt.sum(dim=1)
            else:
                # pos_pt_proxy = scores_gt.sum(dim=1)
                mask = scores < self.config.coarse_matching_thres
                scores[mask] = 0
                pos_pt_proxy = scores.sum(dim=1)
            pos_pt_proxy = pos_pt_proxy > 0
            # <------ select the candidate point sets ------>
            b_idx, n_idx = torch.where(pos_pt_proxy)
            ## <------ randomly sample some coarse-matching due to limited GPU-memory ------>
            # if self.training:
            #     sampled_idx = np.random.permutation(b_idx.shape[0])[0: 64]
            #     b_idx = b_idx[sampled_idx]
            #     n_idx = n_idx[sampled_idx]
            # else:
            #     pass

            pt2ptproxy, sortd_idx = torch.sort(pt2ptproxy, dim=1)
            pt_num_at_proxy = torch_scatter.scatter_sum(torch.ones(pt2ptproxy.shape, device=device), pt2ptproxy, dim=1)
            b, pt_proxy_num = pt_num_at_proxy.shape
            sample_idx = torch.arange(self.pt_sample_num).to(device)
            sample_idx = sample_idx.unsqueeze(0).unsqueeze(0).repeat(b, pt_proxy_num, 1)
            sample_pt_mask = sample_idx < pt_num_at_proxy.unsqueeze(-1)
            sample_idx = sample_idx % pt_num_at_proxy.unsqueeze(-1)
            idx_base = torch.cumsum(pt_num_at_proxy, dim=1)
            tail_base = idx_base[:, 0:-1].clone()
            idx_base[:,1:] = tail_base
            idx_base[:,0] = 0
            sample_idx = sample_idx + idx_base.unsqueeze(-1)
            sample_idx = sample_idx.view(b, -1).long()

            point_xy = data_batch['point_xy_float'].to(device)
            sorted_point_xy = self.index_feats(point_xy, sortd_idx).permute(0, 2, 1)
            sampled_point_xy = self.index_feats(sorted_point_xy, sample_idx).permute(0, 2, 1)
            sampled_point_xy = sampled_point_xy.view(b, -1, pt_proxy_num, self.pt_sample_num)
            sampled_point_xy = sampled_point_xy[b_idx, :, n_idx, :]

            sampled_scores = scores[b_idx, :, n_idx]
            data_batch['sampled_coarse_scores'] = sampled_scores

        if self.training:
            pass
        else:
            point_3d = data_batch['pc']
            sorted_point_3d = self.index_feats(point_3d, sortd_idx).permute(0, 2, 1)
            sampled_point_3d = self.index_feats(sorted_point_3d, sample_idx).permute(0, 2, 1)
            sampled_point_3d = sampled_point_3d.view(b, -1, pt_proxy_num, self.pt_sample_num)
            sampled_point_3d = sampled_point_3d[b_idx, :, n_idx, :]
            data_batch['sampled_point_3d'] = sampled_point_3d

        sorted_pt_feat = self.index_feats(fused_pt_feat, sortd_idx).permute(0, 2, 1)
        sampled_point_feat = self.index_feats(sorted_pt_feat, sample_idx).permute(0,2,1)
        sampled_point_feat = sampled_point_feat.view(b, -1, pt_proxy_num, self.pt_sample_num)
        sampled_point_feat = sampled_point_feat[b_idx, :, n_idx, :]
        sample_pt_mask = sample_pt_mask[b_idx, n_idx, :]

        # <====== sample the corresponding top-k image patches for each point proxy ======>
        with torch.no_grad():
            x = torch.arange(self.config.image_W, dtype=torch.long).cuda()
            y = torch.arange(self.config.image_H, dtype=torch.long).cuda()
            x = x.unsqueeze(0).unsqueeze(0).repeat(b, self.config.image_H, 1)
            y = y.unsqueeze(0).unsqueeze(-1).repeat(b, 1, self.config.image_W)
            pixel_xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).view(b, 2, -1)
            if self.training:
                num_map = data_batch['num_map']
                # pos_img_proxy = scores_gt * pos_pt_proxy.unsqueeze(1)
                pos_img_proxy = num_map * pos_pt_proxy.unsqueeze(1)
            else:
                # num_map = data_batch['num_map']
                # pos_img_proxy = num_map * pos_pt_proxy.unsqueeze(1)
                pos_img_proxy = scores * pos_pt_proxy.unsqueeze(1)
                # pos_img_proxy = scores_gt * pos_pt_proxy.unsqueeze(1)
            rand_img_proxy = torch.randint(0, self.img_proxy_num, (b, self.config.topk_proxy, pt_proxy_num)).to(device)
            v, top_idx = torch.topk(pos_img_proxy, k=self.config.topk_proxy, dim=1)
            # scio.savemat("pos.mat", {"pos_img_proxy": pos_img_proxy.cpu().numpy(), "cs": scores_gt.cpu().numpy()})
            mask_v = v > 0
            top_idx[~mask_v] = rand_img_proxy[~mask_v]
            mask_v = mask_v.unsqueeze(1).repeat(1,int(self.config.patch_size**2),1,1).permute(0,3,2,1).contiguous()
            sample_pixel_mask = mask_v.view(mask_v.shape[0], mask_v.shape[1], -1)

            pixel_idx = self.calculate_pixel_idx(top_idx)
            # scio.savemat("idx.mat", {"top_idx": top_idx[0].cpu().numpy(),"pixel_idx": pixel_idx[0].cpu().numpy()})
            pixel_num_in_patch = pixel_idx.shape[2]
            pixel_idx = pixel_idx.view(b,-1)
            sampled_pixel_xy = self.index_feats(pixel_xy, pixel_idx).permute(0, 2, 1)
            sampled_pixel_xy = sampled_pixel_xy.view(b, 2, pt_proxy_num, pixel_num_in_patch)
            sampled_pixel_xy = sampled_pixel_xy[b_idx, :, n_idx, :]

            sample_pixel_mask = sample_pixel_mask[b_idx, n_idx, :]

        dim_fused = fused_img_feat.shape[1]
        fused_img_feat = fused_img_feat.view(b, dim_fused, -1)
        sampled_pixel_feat = self.index_feats(fused_img_feat, pixel_idx).permute(0, 2, 1)
        sampled_pixel_feat = sampled_pixel_feat.view(b, dim_fused, pt_proxy_num, pixel_num_in_patch)
        sampled_pixel_feat = sampled_pixel_feat[b_idx, :, n_idx, :]

        # <------ construct fine ground-truth ------>
        with torch.no_grad():
            dist = torch.norm(sampled_point_xy.unsqueeze(-2) - sampled_pixel_xy.unsqueeze(-1), p=2, dim=1, keepdim=False)
            fine_gt = torch.zeros_like(dist)
            mask = dist <= self.config.fine_dist_theshold
            fine_gt[mask] = 1#(-1 * dist[mask] / self.config.fine_dist_theshold).exp()
            # slack_row = torch.clamp(1. - torch.sum(fine_gt, dim=-1), min=0.).unsqueeze(-1)
            # slack_col = torch.clamp(1. - torch.sum(fine_gt, dim=-2), min=0.).unsqueeze(-2)
            # supp = torch.zeros(size=(fine_gt.shape[0], 1, 1), dtype=torch.float32).to(device)
            # slack_col = torch.cat([slack_col, supp], dim=-1)
            #
            # fine_gt = torch.cat([fine_gt, slack_row], dim=-1)
            # fine_gt = torch.cat([fine_gt, slack_col], dim=-2)
            # scio.savemat("fine_gt.mat", {"fine_gt": fine_gt.cpu().numpy(),"sample_pt_mask": sample_pt_mask.cpu().numpy(),"sample_pixel_mask": sample_pixel_mask.cpu().numpy()})
            # fine_gt = fine_gt[:,:-1,:-1]
            fine_gt = fine_gt * sample_pt_mask.unsqueeze(1)
            fine_gt = fine_gt * sample_pixel_mask.unsqueeze(2)
            slack_row = torch.clamp(1. - torch.sum(fine_gt, dim=-1), min=0.).unsqueeze(-1)
            slack_col = torch.clamp(1. - torch.sum(fine_gt, dim=-2), min=0.).unsqueeze(-2)
            supp = torch.zeros(size=(fine_gt.shape[0], 1, 1), dtype=torch.float32).to(device)
            slack_col = torch.cat([slack_col, supp], dim=-1)

            fine_gt = torch.cat([fine_gt, slack_row], dim=-1)
            fine_gt = torch.cat([fine_gt, slack_col], dim=-2)

        # <------ masked attention to extract interactive features between resampled points and pixels ------>
        sampled_pixel_feat = sampled_pixel_feat.permute(0, 2, 1)
        sampled_point_feat = sampled_point_feat.permute(0, 2, 1)
        for i in range(self.config.num_ca_layer_fine):
            # <------cross-attention from point to pixel------>
            layer = self.p2i_ca_layers[i]
            sampled_pixel_feat = layer(sampled_pixel_feat, sampled_point_feat, sample_pixel_mask, sample_pt_mask)
            # <------cross-attention from pixel to point------>
            layer = self.i2p_ca_layers[i]
            sampled_point_feat = layer(sampled_point_feat, sampled_pixel_feat, sample_pt_mask, sample_pixel_mask)
            # <------self-attention------>
            layer = self.img_sa_layers[i]
            sampled_pixel_feat = layer(sampled_pixel_feat, sampled_pixel_feat, sample_pixel_mask, sample_pixel_mask)
            layer = self.pt_sa_layers[i]
            sampled_point_feat = layer(sampled_point_feat, sampled_point_feat, sample_pt_mask, sample_pt_mask)

        # masked pair-wise distance matrix
        dim = sampled_pixel_feat.size(2)
        fine_scores = torch.einsum('bnd,bmd->bnm', sampled_pixel_feat, sampled_point_feat)
        fine_scores = fine_scores / dim ** 0.5
        feat_dist = fine_scores
        fine_scores[(~sample_pixel_mask.unsqueeze(2)).repeat(1,1,fine_scores.shape[2])] = -1e6
        fine_scores[(~sample_pt_mask.unsqueeze(1)).repeat(1,fine_scores.shape[1],1)] = -1e6

        # <------Sinkhorn optimal transport------>
        fine_scores = self.optimal_transport(fine_scores)

        fine_loss = self.fine_matching_loss(fine_scores, fine_gt)

        data_batch['sampled_pixel_xy'] = sampled_pixel_xy
        data_batch['sampled_point_xy'] = sampled_point_xy
        data_batch['sample_pixel_mask'] = sample_pixel_mask
        data_batch['sample_pt_mask'] = sample_pt_mask
        data_batch['feat_dist'] = feat_dist
        data_batch['fine_scores'] = fine_scores.exp()
        data_batch['fine_gt'] = fine_gt
        data_batch['fine_loss'] = fine_loss

        return 0
