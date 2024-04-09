import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import random
import argparse
from config import KittiConfiguration, NuScenesConfiguration
from dataset import KittiDataset, NuScenesDataset
from models import CoarseI2P


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Image to point Registration')
    parser.add_argument('--dataset', type=str, default='kitti', help=" 'kitti' or 'nuscenes' ")
    args = parser.parse_args()

    # <------Configuration parameters------>
    if args.dataset == "kitti":
        config = KittiConfiguration()
        train_dataset = KittiDataset(config, mode='train')
        val_dataset = KittiDataset(config, mode='val')
    elif args.dataset == "nuscenes":
        config = NuScenesConfiguration()
        train_dataset = NuScenesDataset(config, mode='train')
        val_dataset = NuScenesDataset(config, mode='val')
    else:
        assert False, "No this dataset choice. Please configure your custom dataset first!"

    set_seed(config.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.c_train_batch_size, shuffle=True,
                                              drop_last=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.c_val_batch_size, shuffle=False,
                                             drop_last=False, num_workers=config.num_workers)

    model = CoarseI2P(config)
    model = model.cuda()

    if config.c_resume:
        assert config.c_checkpoint is not None, "Resume checkpoint error, please set a checkpoint in configuration file!"
        sate_dict = torch.load(config.c_checkpoint)
        model.load_state_dict(sate_dict)
    else:
        print("New Training!")

    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.c_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.c_lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )

    if config.lr_scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_gamma,
        )
    elif config.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.c_step_size,
            gamma=config.c_scheduler_gamma,
        )
    elif config.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min = 0.0001,
        )

    now_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    log_dir = os.path.join(config.logdir, args.dataset + "_"  + str(config.num_pt) + "_coarse_" + now_time)
    ckpt_dir = os.path.join(config.ckpt_dir, args.dataset + "_"  + str(config.num_pt) + "_coarse_" + now_time)
    if os.path.exists(ckpt_dir):
        pass
    else:
        os.makedirs(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    pre_best_coarse_loss = 1e7

    model.train()
    for epoch in range(config.c_epoch):
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        for data in tqdm(train_loader):
            # <------------- validation -------------->
            if global_step % config.c_val_interval == 0:
                with torch.no_grad():
                    model.eval()
                    coarse_loss_list = []
                    for v_data in tqdm(val_loader):
                        model(v_data)
                        coarse_loss = v_data['coarse_loss']
                        coarse_loss_list.append(coarse_loss.cpu().numpy())
                    coarse_loss_list = np.array(coarse_loss_list)
                    writer.add_scalar('val_coarse_loss', coarse_loss_list.mean(), global_step=global_step)
                    if coarse_loss_list.mean() < pre_best_coarse_loss:
                        pre_best_coarse_loss = coarse_loss_list.mean()
                        filename = "step-%d-loss-%f.pth" % (global_step, pre_best_coarse_loss)
                        save_path = os.path.join(ckpt_dir, filename)
                        torch.save(model.state_dict(), save_path)
                        torch.save(model.state_dict(), config.ckpt_dir + "coarse.pth")
                    model.train()

            # <---------------- training ----------------->
            optimizer.zero_grad()
            model(data)

            coarse_loss = data['coarse_loss']

            assert coarse_loss > 0

            coarse_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            writer.add_scalar('training_coarse_loss', coarse_loss, global_step=global_step)
            global_step += 1

        print("%d-th epoch end." % (epoch))
        time.sleep(5)
        lr_scheduler.step()
