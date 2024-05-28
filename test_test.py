# coding: utf-8
import os
import argparse
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.backends import cudnn

# defined
from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HAZERD_ROOT, TEST_TEST_ROOT
# from tools.config import OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy_My, MyModel
from datasets import SotsDataset, OHazeDataset, HazeRDDataset, HazyImagesDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage import color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from tools.vif_utils import vif

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from GAN_module import GANLoss
from GAN_module import D

from loss import ciede2000_color_diff

torch.manual_seed(2018)
torch.cuda.set_device(0)

import numpy as np
import cv2



ckpt_path = './ckpt'
exp_name = 'RESIDE_ITS'
# exp_name = 'O-Haze'
# exp_name = 'HazeRD'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument(
        '--snapshot', type=str, default='version9_RESIDE_ITS_25000', help='snapshot to load for testing')
    return parser.parse_args()

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    # 'HazeRD': HAZERD_ROOT,
    'HazyImages': TEST_TEST_ROOT,
}
args = {
    # 'snapshot': 'iter_40000_loss_0.01267_lr_0.000000',
    # 'snapshot': 'iter_40000_loss_0.01658_lr_0.000000',
    # 'snapshot': 'iter_20000_loss_0.05956_lr_0.000000',
    'snapshot': 'version9_RESIDE_ITS_25000',
}  
to_pil = transforms.ToPILImage()

def main():
    with torch.no_grad():
        # netDiscriminator = D().cuda().train()
        # criterion = nn.L1Loss().cuda()
        # criterionGAN = GANLoss().cuda()
        for name, root in to_test.items():
            if 'SOTS' in name:
                net = MyModel().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy_My().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                net = MyModel().cuda()
                dataset = HazeRDDataset(root, 'test')
            elif 'HazyImages' in name:
                net = MyModel().cuda()
                dataset = HazyImagesDataset(root)
            else:
                raise NotImplementedError

            if len(args.snapshot) > 0:
                print('load snapshot \'%s\' for testing' % args.snapshot)
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')))
            print("yes")
            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            for idx, data in enumerate(dataloader):
                print("testing......")
                haze, file_name = data
                haze = haze.cuda()

                res = sliding_forward(net, haze).detach()
                
                for r, f in zip(res.cpu(), file_name):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name, 
                                     '(%s) %s_%s' % (exp_name, name, args.snapshot), '%s_dehazed.png' % f))

                print(f"Processed {file_name}")

if __name__ == '__main__':
    args = parse_args()

    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))
    main()