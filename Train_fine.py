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
from models import FineI2P


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


if __name__ == '__main__':
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.f_train_batch_size, shuffle=True,
                                               drop_last=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.f_val_batch_size, shuffle=False,
                                              drop_last=True, num_workers=config.num_workers)

    model = FineI2P(config)
    model = model.cuda()

    if config.f_resume:
        assert config.f_checkpoint is not None, "Resume checkpoint error, please set a checkpoint in configuration file!"
        sate_dict = torch.load(config.f_checkpoint)
        model.load_state_dict(sate_dict)
    else:
        print("New Training!")

    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.f_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.f_lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )

    if config.lr_scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.f_scheduler_gamma,
        )
    elif config.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.f_step_size,
            gamma=config.f_scheduler_gamma,
        )
    elif config.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=0.0001,
        )

    now_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    log_dir = os.path.join(config.logdir, args.dataset + "_"  + str(config.num_pt) + "_fine_" + now_time)
    ckpt_dir = os.path.join(config.ckpt_dir, args.dataset + "_"  + str(config.num_pt) + "_fine_" + now_time)
    if os.path.exists(ckpt_dir):
        pass
    else:
        os.makedirs(ckpt_dir)
        os.makedirs(ckpt_dir + "/fine/")
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    pre_best_fine_loss = 1e7

    model.train()
    for epoch in range(config.f_epoch):
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        for data in tqdm(train_loader):
            # <------------- validation -------------->
            if global_step % config.f_val_interval == 0:
                with torch.no_grad():
                    model.eval()
                    loss_fine_list = []
                    for v_data in tqdm(val_loader):
                        model(v_data)
                        fine_loss = v_data['fine_loss']
                        loss_fine_list.append(fine_loss.cpu().numpy())
                    loss_fine_list = np.array(loss_fine_list)
                    writer.add_scalar('val_fine_loss', loss_fine_list.mean(), global_step=global_step)

                    x = loss_fine_list.mean()
                    if x < pre_best_fine_loss:
                        pre_best_fine_loss = x if ~np.isnan(x) else pre_best_fine_loss
                        filename = "fine/step-%d-loss-%f.pth" % (global_step, pre_best_fine_loss)
                        save_path = os.path.join(ckpt_dir, filename)
                        torch.save(model.state_dict(), save_path)
                        torch.save(model.state_dict(), config.ckpt_dir + "fine.pth")
                    model.train()

            # <---------------- training ----------------->
            optimizer.zero_grad()

            model(data)

            fine_loss = data['fine_loss']

            loss = fine_loss

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

            writer.add_scalar('fine_loss', fine_loss, global_step=global_step)
            global_step += 1
            # torch.cuda.empty_cache()
        print("%d-th epoch end." % (epoch))
        time.sleep(5)
        lr_scheduler.step()

