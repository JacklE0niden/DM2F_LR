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
from model import DM2FNet, MyModel

from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT, HAZERD_ROOT
from datasets import ItsDataset, SotsDataset, HazeRDDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from tools.change_image import rgb_to_lab, rgb_to_hsv
from torchsummary import summary
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#newly added 损失函数的扩展
from loss import contrast_loss, tone_loss
from loss import ColorConsistencyLoss, LaplacianFilter, DWT_transform
from loss import compute_multiscale_hf_lf_loss_dwt, compute_multiscale_hf_lf_loss_lp
from GAN_module import D, GANLoss,Discriminator, ImagePool


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 40000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_D': 5e-4,
    # 'lr_discriminator': 0.0001,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 5000,
    # 'val_freq': 3,
    'crop_size': 256
}


def main():
    # net = DM2FNet().cuda().train()
    netDiscriminator = D().cuda().train()
    net = MyModel().cuda().train()
    # net = nn.DataParallel(net)
    summary(net, input_size=(3, 256, 256))
    summary(netDiscriminator, input_size=(3, 256, 256))

    params = list(netDiscriminator.parameters())
    optimizer_Discriminator = torch.optim.Adam(params, lr=cfgs['lr_D'], betas=(0.95, 0.999))

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

    train(net, netDiscriminator, optimizer, optimizer_Discriminator)

def train(net, netDiscriminator, optimizer, optimizer_Discriminator):
    curr_iter = cfgs['last_iter']
    # color_consistency_loss_fn = ColorConsistencyLoss(weight=1.0)  # 实例化 ColorConsistencyLoss
    laplacian_filter = LaplacianFilter()  # 实例化拉普拉斯算子
    dwt_transform = DWT_transform(in_channels=3, out_channels=64).to('cuda')
    while curr_iter <= cfgs['iter_num']:
        # Initialize loss recorders
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()
        # loss_color_record = AvgMeter()
        loss_GAN_record = AvgMeter()
        loss_D_record = AvgMeter()

        hf_lf_loss_record = AvgMeter()

        for data in train_loader:
            # Update learning rates
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']

            haze, gt_trans_map, gt_ato, gt, _ = data
            batch_size = haze.size(0)

            # Move data to GPU
            haze, gt_trans_map, gt_ato, gt = haze.cuda(), gt_trans_map.cuda(), gt_ato.cuda(), gt.cuda()

            # Forward pass through the network
            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)

            criterionGAN = GANLoss(device=haze.device)
            fake_pool = ImagePool(pool_size=0)

            # Discriminator loss for fake samples
            loss_D_fake = 0
            for x_j in [x_jf, x_j0, x_j1, x_j2, x_j3, x_j4]:
                fake_query = fake_pool.query(x_j.detach())
                pred_fake_pool = netDiscriminator.forward(fake_query)
                loss_D_fake += criterionGAN(pred_fake_pool, False)

            # Discriminator loss for real samples
            pred_real = netDiscriminator.forward(gt.detach())
            loss_D_real = criterionGAN(pred_real, True)
            
            # Total discriminator loss
            loss_D = (loss_D_fake + loss_D_real * 6) / 12

            # Update discriminator parameters
            optimizer_Discriminator.zero_grad()
            loss_D.backward()
            optimizer_Discriminator.step()

            # Compute various losses
            loss_x_jf = criterion(x_jf, gt)
            loss_x_j0 = criterion(x_j0, gt)
            loss_x_j1 = criterion(x_j1, gt)
            loss_x_j2 = criterion(x_j2, gt)
            loss_x_j3 = criterion(x_j3, gt)
            loss_x_j4 = criterion(x_j4, gt)
            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)
            # color_loss = 6 * color_consistency_loss_fn(x_jf, gt)
        
            lp_loss_rgb = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, x_jf, criterion, laplacian_filter)
            dwt_loss_rgb = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, x_jf, criterion, dwt_transform)  
            
            lp_loss_lab = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, rgb_to_lab(x_jf), criterion, laplacian_filter)
            dwt_loss_lab = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, rgb_to_lab(x_jf), criterion, dwt_transform)  

            lp_loss_hsv = 0.3 * compute_multiscale_hf_lf_loss_lp(gt, rgb_to_hsv(x_jf), criterion, laplacian_filter)
            dwt_loss_hsv = 0.3 * compute_multiscale_hf_lf_loss_dwt(gt, rgb_to_hsv(x_jf), criterion, dwt_transform)  

            hf_lf_loss = lp_loss_rgb + dwt_loss_rgb + lp_loss_lab + dwt_loss_lab + lp_loss_hsv + dwt_loss_hsv
            hf_lf_loss = lp_loss_rgb + dwt_loss_rgb + lp_loss_lab + dwt_loss_lab + lp_loss_hsv + dwt_loss_hsv
            # 
            # Total generator loss
            loss = (loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 +
                    10 * loss_t + loss_a + hf_lf_loss)
            
            # Generator adversarial loss
            loss_G_GAN = 0
            for x_j in [x_jf, x_j0, x_j1, x_j2, x_j3, x_j4]:
                pred_fake = netDiscriminator(x_j)
                loss_G_GAN += criterionGAN(pred_fake, True)
            loss_G_GAN /= 6
            # 
            # for p in netDiscriminator.parameters():
            #     p.requires_grad=False
            
            optimizer.zero_grad()
            loss_G_GAN.backward(retain_graph=True)

            loss.backward()

            optimizer.step()

            # update recorder
            train_loss_record.update(loss.item(), batch_size)
            #newly added GAN损失
            loss_D_record.update(loss_D.item(), batch_size)
            loss_GAN_record.update(loss_G_GAN.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)

            loss_t_record.update(loss_t.item(), batch_size)
            # loss_color_record.update(color_loss.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)

            hf_lf_loss_record.update(hf_lf_loss.item(), batch_size)

            curr_iter += 1
            
            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f],[loss_t %.5f], [loss_a %.5f], ' \
                '[loss_D %.5f], [loss_GAN %.5f],[hf_lf_loss %.5f],' \
                '[lr %.13f]' % \
                (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                 loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg, 
                 loss_t_record.avg, loss_a_record.avg,
                 loss_D_record.avg, loss_GAN_record.avg, hf_lf_loss_record.avg,
                 optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()

class TransLoss(nn.Module):
    def __init__(self):
        super(TransLoss, self).__init__()

    def forward(self, test_output, target_output):
        # Ensure there are no zero values to avoid division by zero
        test_output = torch.clamp(test_output, min=1e-10)
        target_output = torch.clamp(target_output, min=1e-10)
        
        # Compute the custom loss
        loss = torch.abs((1/test_output - 1/target_output)*1e-10).mean()
        return loss


if __name__ == '__main__':
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=8)

    criterion = nn.L1Loss().cuda()
    transloss = TransLoss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
