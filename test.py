# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

# defined
# from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT
from tools.config import OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color
from tools.vif_utils import vif
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
exp_name = 'O-Haze'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    'snapshot': 'iter_20000_loss_0.05956_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                print("dm2fnet")
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                print("dm2fnet_wophy")
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'test')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, vifs, niqe = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])


                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    # ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                    #                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    # newlyadded
                    ssim = structural_similarity(gt, r, win_size=3, data_range=1, multichannel=True,
                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)                    

                    ssims.append(ssim)

                    gt_gray = color.rgb2gray(gt)
                    r_gray = color.rgb2gray(r)
                    
                    # Calculate VIF
                    vif_score = vif(gt_gray, r_gray)
                    vifs.append(vif_score)
                    
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, VIF {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, vif_score))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, VIF: {np.mean(vifs):.6f}")


if __name__ == '__main__':
    main()
