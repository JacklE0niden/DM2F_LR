import torch
import torch.nn as nn
import functools
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
# 这个判别器会爆显存
# class Discriminator(nn.Module):
#     def __init__(self, norm_layer, use_sigmoid, input_nc, ndf):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1),
#             norm_layer(ndf),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf, ndf * 2, kernel_size=3, padding=1),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, padding=1),
#             norm_layer(ndf * 4),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=2, padding=1),
#             norm_layer(ndf * 4),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, padding=1),
#             norm_layer(ndf * 8),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 8, ndf * 8, kernel_size=3, stride=2, padding=1),
#             norm_layer(ndf * 8),
#             nn.LeakyReLU(0.2),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(ndf * 8, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1, kernel_size=1)
#         )
#         self.use_sigmoid = use_sigmoid

#     def forward(self, x):
#         batch_size = x.size(0)
#         if self.use_sigmoid:
#             return torch.sigmoid(self.net(x).view(batch_size))
#         else:
#             return self.net(x).view(batch_size)
    
import random

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
    
    
class Discriminator(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Discriminator, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features
        
        # 冻结预训练的 VGG19 参数
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.use_sigmoid = use_sigmoid

        # 添加自定义的分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x).view(x.size(0))
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def D(use_sigmoid=False, gpu_ids=[]):
    netD = Discriminator(use_sigmoid=use_sigmoid)

    print(netD)
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, device="cuda:0"):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label).to(self.device)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label).to(self.device)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real).to(self.device)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).to(self.device)
            return self.loss(input[-1], target_tensor)