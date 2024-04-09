import numpy as np
import math
import torch


class NuScenesConfiguration:
    """
    The configuration to train on NuScenes dataset
    """
    def __init__(self, data_root=None):
        print("Learning on NuScenes...")
        # <----------- dataset configuration ---------->
        self.dataset_root = '../nuscenes2/' if data_root is None else data_root
        self.num_pt = 40960
        self.P_Tx_amplitude = 10.0
        self.P_Ty_amplitude = 0.0
        self.P_Tz_amplitude = 10.0
        self.P_Rx_amplitude = 0.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0
        self.cropped_img_H = 160
        self.cropped_img_W = 320
        # <-------------------------------------------->

        # <--------- training and testing configuration ---------->
        self.seed = 2023

        # <--------- coarse matching ------->
        self.c_train_batch_size = 8
        self.c_val_batch_size = 8
        self.c_val_interval = 500
        self.c_epoch = 30
        self.c_lr = 0.001
        self.c_resume = False
        self.c_checkpoint = None

        # <-------- fine matching -------->
        self.f_train_batch_size = 4
        self.f_val_batch_size = 8
        self.f_val_interval = 500
        self.f_epoch = 3
        self.f_lr = 0.001
        self.f_resume = False
        self.f_checkpoint = None
        self.used_coarse_checkpoint = "checkpoint/coarse.pth"

        self.num_workers = 16

        # <-------- optimizer -------->
        self.optimizer = "ADAM"  # "SGD" or "ADAM"
        self.momentum = 0.98
        self.weight_decay = 1e-06

        # <-------- lr_scheduler -------->
        self.lr_scheduler = "StepLR"
        self.c_scheduler_gamma = 0.8
        self.f_scheduler_gamma = 0.6
        self.c_step_size = 3
        self.f_step_size = 1

        self.logdir = "log/"
        self.ckpt_dir = "checkpoint/"

        # <-----------model configuration---------->
        # <---------image ViT-------->
        self.image_H = int(self.cropped_img_H * 0.25)
        self.image_W = int(self.cropped_img_W * 0.25)
        self.patch_size = 8
        self.use_resnet_embedding = True

        self.embed_dim = 64
        self.mlp_dim = 1024

        self.embed_dropout = 0.1
        self.mlp_dropout = 0.1
        self.attention_dropout = 0.1

        self.num_sa_layer = 3
        self.num_head = 8
        # <---------point ViT-------->
        self.use_gnn_embedding = False
        self.point_feat_dim = 3  # coordinate + intensity + normal

        if self.use_gnn_embedding:
            self.num_node = 256
            self.edge_conv_dim = 64
        else:
            self.num_node = 1280
            self.num_proxy = 256
        # <---------Coarse I2P-------->
        self.num_ca_layer_coarse = 6
        self.sinkhorn_iters = 100
        self.coarse_matching_thres = 0.01

        # <------ Fine I2P ------>
        self.num_ca_layer_fine = 3
        self.pt_sample_num = 65
        self.fine_dist_theshold = 1
        self.topk_proxy = 3
        self.pixel_positional_embedding = True
        self.fine_loss_weight = 0.5

        self.img_fuse_res_num = 2
        self.pixel_pos_embed_convs_res_num = 3
        self.node_fuse_res_num = 2
        self.pt_head_res_num = 3
        self.linear_attention_num = 4
        self.LA_head_num = 8


