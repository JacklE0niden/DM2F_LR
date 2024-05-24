# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from model import DM2FNet_woPhy_My
from tools.config import OHAZE_ROOT
from datasets import OHazeDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward

#newly added 
from tools.change_image import rgb_to_lab, rgb_to_hsv

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#newly added 损失函数的扩展
from loss import contrast_loss, tone_loss
from loss import ColorConsistencyLoss, LaplacianFilter, DWT_transform
from loss import compute_multiscale_hf_lf_loss_dwt, compute_multiscale_hf_lf_loss_lp

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='.\ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='O-Haze',
        help='experiment name.')
    args = parser.parse_args()

    return args

# 配置信息
cfgs = {
    'use_physical': True,
    'iter_num': 20000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 2e-4,
    'lr_decay': 0.9,
    'weight_decay': 2e-5,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 2000,
    'crop_size': 512,
}


def main():
    net = DM2FNet_woPhy_My().cuda().train()
    # net = DataParallel(net)

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')
    # modify the log path
    # with open(log_path, 'w') as f:
    #     f.write(str(cfgs) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = cfgs['last_iter']
    color_consistency_loss_fn = ColorConsistencyLoss(weight=1.0)  # 实例化 ColorConsistencyLoss
    laplacian_filter = LaplacianFilter()  # 实例化 LaplacianFilter
    dwt_transform = DWT_transform(in_channels=3, out_channels=64).to('cuda')
    scaler = amp.GradScaler()
    torch.cuda.empty_cache()
    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record = AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        
        # newly added 三种损失
        color_loss_record = AvgMeter()
        hf_lf_loss_record = AvgMeter()


        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']

            haze, gt, _ = data

            batch_size = haze.size(0)

            haze, gt = haze.cuda(), gt.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                x_jf, x_j1, x_j2, x_j3, x_j4 = net(haze)

                loss_x_jf = criterion(x_jf, gt)
                loss_x_j1 = criterion(x_j1, gt)
                loss_x_j2 = criterion(x_j2, gt)
                loss_x_j3 = criterion(x_j3, gt)
                loss_x_j4 = criterion(x_j4, gt)

                color_loss = color_consistency_loss_fn(x_jf, gt)
                # 原始图片，在小波变换的高低频损失
                lp_loss_rgb = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, x_jf, criterion, laplacian_filter)
                dwt_loss_rgb = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, x_jf, criterion, dwt_transform)  
                
                lp_loss_lab = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, rgb_to_lab(x_jf), criterion, laplacian_filter)
                dwt_loss_lab = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, rgb_to_lab(x_jf), criterion, dwt_transform)  

                lp_loss_hsv = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, rgb_to_hsv(x_jf), criterion, laplacian_filter)
                dwt_loss_hsv = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, rgb_to_hsv(x_jf), criterion, dwt_transform)  

                hf_lf_loss = lp_loss_rgb + dwt_loss_rgb + lp_loss_lab + dwt_loss_lab + lp_loss_hsv + dwt_loss_hsv

                loss = loss_x_jf + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 + color_loss + hf_lf_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)
            # newly added
            color_loss_record.update(color_loss.item(), batch_size)
            hf_lf_loss_record.update(hf_lf_loss.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [color_loss %.5f], [hf_lf_loss %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_jf_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   color_loss_record.avg, hf_lf_loss_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == 1 or (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)
                torch.cuda.empty_cache()

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnr_record, ssim_record = AvgMeter(), AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data
            haze, gt = haze.cuda(), gt.cuda()

            dehaze = sliding_forward(net, haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1]
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(g, r)
                # ssim = structural_similarity(g, r, data_range=1, multichannel=True,
                #                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                # newlyadded
                ssim = structural_similarity(g, r, win_size=3, data_range=1, multichannel=True,
                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                
                psnr_record.update(psnr)
                ssim_record.update(ssim)

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    log = '[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}]'.format(
        curr_iter + 1, loss_record.avg, psnr_record.avg, ssim_record.avg)
    print(log)
    open(log_path, 'a').write(log + '\n')
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()
    print("args parsed.......")
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    print("gpus = ", args.gpus)
    torch.cuda.set_device(int(args.gpus))
    
    print('O-HAZE_ROOT:', OHAZE_ROOT)
    train_dataset = OHazeDataset(OHAZE_ROOT, 'train_crop_512') # 这里设置为OHAZE数据集的路径，要使用512*512的裁剪(preprocess_ohaze_data.py)
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=1,
                              shuffle=True, drop_last=True)

    val_dataset = OHazeDataset(OHAZE_ROOT, 'val')
    val_loader = DataLoader(val_dataset, batch_size=1)

    criterion = nn.L1Loss().cuda()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 移除最后三位微秒
    filename = timestamp + '.txt'
    # modify the log path
    # 使用 os.path.join() 来构建文件路径
    log_path = os.path.join(args.ckpt_path, args.exp_name, filename)
    # log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')
    print('log path:', log_path)

    main()
