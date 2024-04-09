import torch
import torch.nn as nn
import torch_scatter
import numpy as np
import math
import time
import sys
import scipy.io as scio
from .PointViT import PointTransformer
from .ImageViT import ImageTransformer
sys.path.append("..")
from utils import log_optimal_transport


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
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

    def forward(self, x_hidden_states, y_hidden_states):
        mixed_query_layer = self.query(x_hidden_states)
        mixed_key_layer = self.key(y_hidden_states)
        mixed_value_layer = self.value(y_hidden_states)

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
        self.attn = Attention(config)

    def forward(self, x, y):
        h = x
        x = self.attention_norm(x)
        y = self.attention_norm(y)
        x = self.attn(x, y)
        x = h + x

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + x
        return x


class CoarseI2P(nn.Module):
    """
        Coarse part of CFI2P
    """
    def __init__(self, config):
        super(CoarseI2P, self).__init__()
        self.config = config

        # proxy generation is placed in the two modules
        self.pt_transformer = PointTransformer(config)
        self.img_transformer = ImageTransformer(config)

        # self-attention and cross-attention to capture cross-modal context
        self.i2p_ca_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.i2p_ca_layers.append(Block(config))

        self.p2i_ca_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.p2i_ca_layers.append(Block(config))

        self.pt_sa_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.pt_sa_layers.append(Block(config))

        self.img_sa_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.img_sa_layers.append(Block(config))

        # the slack entry for the optimal transport module
        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def optimal_transport(self, scores):
        return log_optimal_transport(scores, None, None, self.bin_score, iters=self.config.sinkhorn_iters)

    def coarse_matching_loss(self, scores, scores_gt):
        y = scores_gt * scores
        loss = torch.sum(-y, dim=(1, 2)) / torch.sum(scores_gt, dim=(1, 2))
        loss = loss.sum() / scores.shape[0]
        return loss

    def calculate_coarse_ground_truth(self, data_batch):
        with torch.no_grad():
            W = self.config.image_W // self.config.patch_size
            H = self.config.image_H // self.config.patch_size
            pt_xy = data_batch['point_xy'].cuda()
            pc_mask = data_batch['pc_mask'].cuda()
            node2proxy = data_batch['node2proxy']
            pt2node = data_batch['pt2node'].cuda()

            B = pt_xy.shape[0]
            scores_gt = torch.zeros(B, H * W + 1, self.config.num_proxy + 1).cuda()
            num_map = torch.zeros(B, H * W, self.config.num_proxy).cuda()
            for i in range(B):
                xy = pt_xy[i]
                in_mask = pc_mask[i]
                in_xy = xy[:, in_mask]
                in_xy = in_xy // self.config.patch_size

                # <------ the index of point to img-proxy ------>
                inpt2imgproxy = in_xy[1, :] * W + in_xy[0, :]

                # <------  the number of points in each img patch ------>
                inpt2imgproxy_max = torch.tensor([H * W - 1]).long().cuda()
                inpt2imgproxy = torch.cat([inpt2imgproxy, inpt2imgproxy_max])
                num_per_img_proxy = torch_scatter.scatter_sum(torch.ones(inpt2imgproxy.shape,\
                                                                         device=inpt2imgproxy.device), inpt2imgproxy)
                num_per_img_proxy[H * W - 1] -= 1
                inpt2imgproxy = inpt2imgproxy[:-1]

                # <------ the number of points in each point set ------>
                n2p = node2proxy[i]
                pt2n = pt2node[i]
                num_per_pt_node = torch_scatter.scatter_sum(torch.ones(pt2n.shape, device=pt2n.device), pt2n)
                num_per_pt_proxy = torch_scatter.scatter_sum(num_per_pt_node, n2p)

                # <------ the index from in-camera point to point-proxy ------>
                pt2ptproxy = torch.gather(n2p, index=pt2n, dim=0)
                inpt2ptproxy = pt2ptproxy[in_mask]

                # <------ create the ground truth weight matrix ------>
                inpt2map = inpt2imgproxy * self.config.num_proxy + inpt2ptproxy
                inpt2map_max = torch.tensor([H * W * self.config.num_proxy - 1]).long().cuda()
                inpt2map = torch.cat([inpt2map, inpt2map_max])
                num_pt_map = torch_scatter.scatter_sum(torch.ones(inpt2map.shape, device=inpt2map.device), inpt2map)
                num_pt_map[H * W * self.config.num_proxy - 1] -= 1
                num_pt_map = num_pt_map.reshape(H * W, self.config.num_proxy)

                zero_mask = (num_per_img_proxy == 0)
                zero_mask_e = zero_mask.unsqueeze(1).expand(H * W, self.config.num_proxy)
                r_map_0 = num_pt_map / num_per_img_proxy.unsqueeze(1)
                r_map_0[zero_mask_e] = 0

                r_map_1 = num_pt_map / num_per_pt_proxy.unsqueeze(0)

                r_map = torch.where(r_map_0 < r_map_1, r_map_0, r_map_1)
                # scio.savemat("num_pt_map.mat", {"map": num_pt_map.cpu().numpy(), "num_per_pt_proxy":num_per_pt_proxy.cpu().numpy(),"num_per_img_proxy":num_per_img_proxy.cpu().numpy()})
                # print("num_pt_map saved!")
                # time.sleep(100)

                sum_r_map_0 = num_pt_map.sum(dim=0, keepdim=True) / num_per_pt_proxy.unsqueeze(0)
                sum_r_map_1 = num_pt_map.sum(dim=1, keepdim=True) / num_per_img_proxy.unsqueeze(1)
                sum_r_map_1[zero_mask.unsqueeze(1)] = 0

                assert ((sum_r_map_0 > 1).sum() == 0 and (sum_r_map_1 > 1).sum() == 0 and (sum_r_map_0 < 0).sum() == 0 and (sum_r_map_1 < 0).sum() == 0)

                row_coupling = 1 - sum_r_map_0
                col_coupling = 1 - sum_r_map_1

                r_map = torch.cat([r_map, row_coupling], dim=0)
                zeros_0_0 = torch.tensor([0.0]).cuda().unsqueeze(0)
                col_coupling_plus_0 = torch.cat([col_coupling, zeros_0_0], dim=0)
                r_map = torch.cat([r_map, col_coupling_plus_0], dim=1)
                scores_gt[i, :, :] = r_map[:, :]
                num_map[i, :, :] = num_pt_map[:, :]
        return scores_gt, num_map

    def forward(self, data_batch):
        img = data_batch['img'].cuda()
        pc = data_batch['pc'].cuda()
        node = data_batch['node'].cuda()
        idx = data_batch['pt2node'].cuda()

        # <------ pixel proxy generation, learn the global context via Vision Transformer ------>
        img_proxy, img_feat = self.img_transformer(img)
        data_batch['img_feat'] = img_feat

        # <------ point proxy generation, learn the global context via Point Transformer ------>
        # <------ we will sample point proxies from nodes again (2-level hierarchical architecture) ------>
        pt_proxy, node_proxy_idx, pt_feat, node_feat = self.pt_transformer(pc, node, idx)
        data_batch['node2proxy'] = node_proxy_idx[:,:,0]
        data_batch['pt_feat'] = pt_feat
        data_batch['node_feat'] = node_feat

        for i in range(self.config.num_ca_layer_coarse):
            # <------cross-attention from point to image------>
            layer = self.p2i_ca_layers[i]
            img_proxy = layer(img_proxy, pt_proxy)
            # <------cross-attention from image to point------>
            layer = self.i2p_ca_layers[i]
            pt_proxy = layer(pt_proxy, img_proxy)
            # <------self-attention------>
            layer = self.img_sa_layers[i]
            img_proxy = layer(img_proxy, img_proxy)
            layer = self.pt_sa_layers[i]
            pt_proxy = layer(pt_proxy, pt_proxy)

        # <------ pairwise distance matrix ------>
        dim = img_proxy.size(2)
        scores = torch.einsum('bnd,bmd->bnm', img_proxy, pt_proxy)
        scores = scores / dim ** 0.5

        # <------ Sinkhorn optimal transport ------>
        scores = self.optimal_transport(scores)

        # scio.savemat("scores.mat", {"scores": scores[:,:-1,:-1].detach().cpu().numpy()})
        # print("~~")
        # time.sleep(100)

        # <------ calculate the coarse ground-truth ------>
        scores_gt, num_map = self.calculate_coarse_ground_truth(data_batch)

        # <------ coarse-level loss ------>
        coarse_loss = self.coarse_matching_loss(scores, scores_gt)

        data_batch['img_proxy'] = img_proxy
        data_batch['pt_proxy'] = pt_proxy
        data_batch['scores'] = scores.exp()
        data_batch['scores_gt'] = scores_gt
        data_batch['num_map'] = num_map # used for fine training
        data_batch['coarse_loss'] = coarse_loss
        data_batch['pc'] = pc
        return 0
