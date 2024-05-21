import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import time
import os
import cv2
import numpy as np
import torchvision.models as models
from resnext import ResNeXt101


# def white_balance(image):
#     # Compute the average R, G, and B values
#     if isinstance(image, torch.Tensor):
#         image = image.detach().cpu().numpy()
#     avg_r = np.mean(image[:, :, 0])
#     avg_g = np.mean(image[:, :, 1])
#     avg_b = np.mean(image[:, :, 2])

#     # Compute the average grayscale value
#     avg_gray = (avg_r + avg_g + avg_b) / 3

#     # Compute the scaling factors for each channel
#     scale_r = avg_gray / avg_r
#     scale_g = avg_gray / avg_g
#     scale_b = avg_gray / avg_b

#     # Apply the scaling factors to each channel
#     balanced_image = np.zeros_like(image, dtype=np.float32)
#     balanced_image[:, :, 0] = np.clip(image[:, :, 0] * scale_r, 0, 255)
#     balanced_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
#     balanced_image[:, :, 2] = np.clip(image[:, :, 2] * scale_b, 0, 255)

#     return balanced_image.astype(np.uint8)

# def contrast_enhance(image):
#     # Compute the mean and standard deviation of each channel
#     if isinstance(image, torch.Tensor):
#         image = image.detach().cpu().numpy()
#     mean_r = np.mean(image[:, :, 0])
#     mean_g = np.mean(image[:, :, 1])
#     mean_b = np.mean(image[:, :, 2])
#     std_r = np.std(image[:, :, 0])
#     std_g = np.std(image[:, :, 1])
#     std_b = np.std(image[:, :, 2])

#     # Apply contrast enhancement using mean and standard deviation
#     enhanced_image = np.zeros_like(image, dtype=np.float32)
#     enhanced_image[:, :, 0] = np.clip((image[:, :, 0] - mean_r) * (128 / std_r) + 128, 0, 255)
#     enhanced_image[:, :, 1] = np.clip((image[:, :, 1] - mean_g) * (128 / std_g) + 128, 0, 255)
#     enhanced_image[:, :, 2] = np.clip((image[:, :, 2] - mean_b) * (128 / std_b) + 128, 0, 255)

#     return enhanced_image.astype(np.uint8)

# def gamma_correction(image, gamma=1.0):
#     # Apply gamma correction to each channel
#     if isinstance(image, torch.Tensor):
#         image = image.detach().cpu().numpy()
#     corrected_image = np.zeros_like(image, dtype=np.float32)
#     corrected_image[:, :, 0] = np.clip(image[:, :, 0] ** gamma, 0, 255)
#     corrected_image[:, :, 1] = np.clip(image[:, :, 1] ** gamma, 0, 255)
#     corrected_image[:, :, 2] = np.clip(image[:, :, 2] ** gamma, 0, 255)

#     return corrected_image.astype(np.uint8)

# def preprocess_image(image):
#     # Ensure image is in the correct format (HWC)
#     if len(image.shape) == 4:  # Assume batch dimension
#         images_wb = []
#         images_ce = []
#         images_gc = []
#         for i in range(image.shape[0]):
#             single_image = image[i]  # Take ith image from batch
#             # print("single_image.shape:", single_image.shape)
#             if len(single_image.shape) != 3 or single_image.shape[0] != 3:
#                 raise ValueError("Invalid image format. Expected RGB image.")
            
#             # Preprocess each single image
#             image_wb = white_balance(single_image)
#             image_ce = contrast_enhance(single_image)
#             image_gc = gamma_correction(single_image)

#             # Append processed images to the list
#             images_wb.append(image_wb)
#             images_ce.append(image_ce)
#             images_gc.append(image_gc)

#         # Convert lists to arrays
#         images_wb = np.array(images_wb)
#         images_ce = np.array(images_ce)
#         images_gc = np.array(images_gc)

#     return images_wb, images_ce, images_gc



class HorizontalPoolingPyramid(): # 水平金字塔池化方法
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            # bin_num = [16, 8, 4, 2, 1]
            bin_num = [8, 4, 2, 1]
        self.bin_num = bin_num
    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1) 
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        rgb_mean = (0.485, 0.456, 0.406)
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.229, 0.224, 0.225)
        self.std = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class BaseA(nn.Module):
    def __init__(self):
        super(BaseA, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63438
        self.mean[0, 1, 0, 0] = 0.59396
        self.mean[0, 2, 0, 0] = 0.58369
        self.std[0, 0, 0, 0] = 0.16195
        self.std[0, 1, 0, 0] = 0.16937
        self.std[0, 2, 0, 0] = 0.17564

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class BaseITS(nn.Module):
    def __init__(self):
        super(BaseITS, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63542
        self.mean[0, 1, 0, 0] = 0.59579
        self.mean[0, 2, 0, 0] = 0.58550
        self.std[0, 0, 0, 0] = 0.14470
        self.std[0, 1, 0, 0] = 0.14850
        self.std[0, 2, 0, 0] = 0.15348

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class Base_OHAZE(nn.Module):
    def __init__(self):
        super(Base_OHAZE, self).__init__()
        rgb_mean = (0.47421, 0.50878, 0.56789)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.10168, 0.10488, 0.11524)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)

        rgb_mean = (0.35851, 0.35316, 0.34425)
        self.mean_out = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.16391, 0.16174, 0.17148)
        self.std_out = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class J0(Base):
    def __init__(self, num_features=128):
        super(J0, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        a = self.a(f)
        t = F.upsample(self.t(f), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0, max=1)

        if self.training:
            return x_phy, t, a.view(x.size(0), -1)
        else:
            return x_phy


class J1(Base):
    def __init__(self, num_features=128):
        super(J1, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))

        p0 = F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        return x_p0


class J2(Base):
    def __init__(self, num_features=128):
        super(J2, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        p1 = F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        return x_p1


class J3(Base):
    def __init__(self, num_features=128):
        super(J3, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        p2 = F.upsample(self.p2(f), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        return x_p2


class J4(Base):
    def __init__(self, num_features=128):
        super(J4, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))

        p3 = F.upsample(self.p3(f), size=x0.size()[2:], mode='bilinear')
        # x_p3 = (torch.log(1 + p3 * x0)).clamp(min=0, max=1)
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        return x_p3


class ours_wo_AFIM(Base):
    def __init__(self, num_features=128):
        super(ours_wo_AFIM, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        f = (down1 + down2 + down3 + down4) / 4
        f = self.refine(f) + f

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        a = self.a(f)
        t = F.upsample(self.t(f), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0, max=1)

        p0 = F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        p1 = F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        p2 = F.upsample(self.p2(f), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        p3 = F.upsample(self.p3(f), size=x0.size()[2:], mode='bilinear')
        # x_p3 = (torch.log(1 + p3 * x0)).clamp(min=0, max=1)
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        attention_fusion = F.upsample(self.attentional_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 5, :, :], 1) * torch.stack(
            (x_phy[:, 0, :, :], x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) * torch.stack((x_phy[:, 1, :, :],
                                                                                                      x_p0[:, 1, :, :],
                                                                                                      x_p1[:, 1, :, :],
                                                                                                      x_p2[:, 1, :, :],
                                                                                                      x_p3[:, 1, :, :]),
                                                                                                     1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) * torch.stack((x_phy[:, 2, :, :],
                                                                                                    x_p0[:, 2, :, :],
                                                                                                    x_p1[:, 2, :, :],
                                                                                                    x_p2[:, 2, :, :],
                                                                                                    x_p3[:, 2, :, :]),
                                                                                                   1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_phy, x_p0, x_p1, x_p2, x_p3, t, a.view(x.size(0), -1)
        else:
            return x_fusion


class ours_wo_J0(Base):
    def __init__(self, num_features=128):
        super(ours_wo_J0, self).__init__()
        self.num_features = num_features

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.attention0 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 12, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean) / self.std

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        n, c, h, w = down1.size()

        attention0 = self.attention0(concat)
        attention0 = F.softmax(attention0.view(n, 4, c, h, w), 1)
        f0 = down1 * attention0[:, 0, :, :, :] + down2 * attention0[:, 1, :, :, :] + \
             down3 * attention0[:, 2, :, :, :] + down4 * attention0[:, 3, :, :, :]
        f0 = self.refine(f0) + f0

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        p0 = F.upsample(self.p0(f0), size=x0.size()[2:], mode='bilinear')
        x_p0 = torch.exp(log_x0 + p0).clamp(min=0, max=1)

        p1 = F.upsample(self.p1(f1), size=x0.size()[2:], mode='bilinear')
        x_p1 = ((x + p1) * self.std + self.mean).clamp(min=0, max=1)

        p2 = F.upsample(self.p2(f2), size=x0.size()[2:], mode='bilinear')
        x_p2 = torch.exp(-torch.exp(log_log_x0_inverse + p2)).clamp(min=0, max=1)

        p3 = F.upsample(self.p3(f3), size=x0.size()[2:], mode='bilinear')
        x_p3 = (torch.log(1 + torch.exp(log_x0 + p3))).clamp(min=0, max=1)

        attention_fusion = F.upsample(self.attentional_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 4, :, :], 1) * torch.stack(
            (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 4: 8, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
                                                                                                     x_p1[:, 1, :, :],
                                                                                                     x_p2[:, 1, :, :],
                                                                                                     x_p3[:, 1, :, :]),
                                                                                                    1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 8:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
                                                                                                   x_p1[:, 2, :, :],
                                                                                                   x_p2[:, 2, :, :],
                                                                                                   x_p3[:, 2, :, :]),
                                                                                                  1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_p0, x_p1, x_p2, x_p3
        else:
            return x_fusion


class DM2FNet(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d'):
        super(DM2FNet, self).__init__()
        self.num_features = num_features

        # resnext = ResNeXt101()
        #
        # self.layer0 = resnext.layer0
        # self.layer1 = resnext.layer1
        # self.layer2 = resnext.layer2
        # self.layer3 = resnext.layer3
        # self.layer4 = resnext.layer4

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=False)
        del backbone.fc
        self.backbone = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        # 用来预测t0

        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )
        # 用来预测A0

        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):
        x = (x0 - self.mean) / self.std
        # print("x.shape", x.shape)

        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        # layer0 = self.layer0(x)
        # layer0.shape: torch.Size([16, 64, 64, 64])
        # layer1.shape: torch.Size([16, 256, 64, 64])
        # layer2.shape: torch.Size([16, 512, 32, 32])
        # layer3.shape: torch.Size([16, 1024, 16, 16])
        # layer4.shape: torch.Size([16, 2048, 8, 8])


        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)
        # 得到不同层级的特征表示（MLF）

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)
        

        # down1.shape: torch.Size([16, 128, 64, 64])
        # down2.shape: torch.Size([16, 128, 64, 64])
        # down3.shape: torch.Size([16, 128, 64, 64])
        # down4.shape: torch.Size([16, 128, 64, 64])
        # concat.shape: torch.Size([16, 512, 64, 64])

        n, c, h, w = down1.size()

        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 4, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :]
        f_phy = self.refine(f_phy) + f_phy # [16,128,64,64]
        # 这个特征是用来预测t的

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1 # [16,128,64,64]

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2 # [16,128,64,64]

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3 # [16,128,64,64]

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
             down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4 # [16,128,64,64]

        # 4个不同的AFIM


        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        # print("f_phy.shape:", f_phy.shape)
        a = self.a(f_phy) # 对a的预测
        # print("a.shape:", a.shape)
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear') # 对t的预测
        # print("t.shape:", t.shape)
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)
        # print("x_phy.shape:",x_phy.shape) # [16, 3, 256, 256]

        # J1 = I * R1
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)
        # print("x_j1.shape:",x_j1.shape) # [16, 3, 256, 256]

        # J2 = I + R2
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)
        # print("x_j2.shape:",x_j2.shape) # [16, 3, 256, 256]

        # J3 = I * exp(R3)
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)
        # print("x_j3.shape:",x_j3.shape) # [16, 3, 256, 256]

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)
        # print("x_j4.shape:",x_j4.shape) # [16, 3, 256, 256]

        attention_fusion = F.upsample(self.attention_fusion(concat), size=x0.size()[2:], mode='bilinear')
        # 那一大坨W0~W4
        # print("attention_fusion.shape:",attention_fusion.shape) # [16, 15, 256, 256]
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                      x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
        # print("x_f0.shape:",x_f0.shape) # [16, 1, 256, 256]

        x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                      x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
        
        # print("x_f1.shape:",x_f1.shape)

        x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                      x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
        # print("x_f2.shape:",x_f2.shape)

        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)
        # print("x_fusion.shape:",x_fusion.shape) # [16, 3, 256, 256]

        if self.training:
            return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t, a.view(x.size(0), -1)
        else:
            return x_fusion


class DM2FNet_woPhy(Base_OHAZE):
    def __init__(self, num_features=64, arch='resnext101_32x8d'):
        super(DM2FNet_woPhy, self).__init__()
        self.num_features = num_features

        # resnext = ResNeXt101Syn()
        # self.layer0 = resnext.layer0
        # self.layer1 = resnext.layer1
        # self.layer2 = resnext.layer2
        # self.layer3 = resnext.layer3
        # self.layer4 = resnext.layer4

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=False)
        del backbone.fc
        self.backbone = backbone
        # print("backbone", self.backbone)
        # backbone ResNet
        self.down0 = nn.Sequential(
            nn.Conv2d(64, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.fuse3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse0 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

        self.fuse3_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse2_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse1_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse0_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 12, kernel_size=3, padding=1)
        )

        # self.vgg = VGGF()

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean_in) / self.std_in
        # 对模糊图像进行标准化处理，x就表示I

        # resnet backbone
        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)
        
        down0 = self.down0(layer0)
        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)
        # 得到不同层级的特征表示（MLF）

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        fuse3_attention = self.fuse3_attention(torch.cat((down4, down3), 1))
        f = down4 + self.fuse3(torch.cat((down4, fuse3_attention * down3), 1))

        f = F.upsample(f, size=down2.size()[2:], mode='bilinear')
        fuse2_attention = self.fuse2_attention(torch.cat((f, down2), 1))
        f = f + self.fuse2(torch.cat((f, fuse2_attention * down2), 1))

        f = F.upsample(f, size=down1.size()[2:], mode='bilinear')
        fuse1_attention = self.fuse1_attention(torch.cat((f, down1), 1))
        f = f + self.fuse1(torch.cat((f, fuse1_attention * down1), 1))

        f = F.upsample(f, size=down0.size()[2:], mode='bilinear')
        fuse0_attention = self.fuse0_attention(torch.cat((f, down0), 1))
        f = f + self.fuse0(torch.cat((f, fuse0_attention * down0), 1))
        # 计算出四个AFIM之后的结果

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))# 对输入的模糊图像 x0 进行了双重截断操作

        x_p0 = torch.exp(log_x0 + F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0, max=1)
        # 上采样到与输入图像相同的大小。计算模糊图像的无雾背景细节层。

        x_p1 = ((x + F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)\
            .clamp(min=0., max=1.)
        # 用来计算模糊图像的辐射层。

        log_x_p2_0 = torch.log(
            ((x + F.upsample(self.p2_0(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)
                .clamp(min=1e-8))
        x_p2 = torch.exp(log_x_p2_0 + F.upsample(self.p2_1(f), size=x0.size()[2:], mode='bilinear'))\
            .clamp(min=0., max=1.)
        # 计算模糊图像的透射层

        log_x_p3_0 = torch.exp(log_log_x0_inverse + F.upsample(self.p3_0(f), size=x0.size()[2:], mode='bilinear'))
        x_p3 = torch.exp(-log_x_p3_0 + F.upsample(self.p3_1(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0,
                                                                                                            max=1)
        # 计算模糊图像的雾层

        attention_fusion = F.upsample(self.attentional_fusion(f), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 4, :, :], 1) * torch.stack(
            (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 4: 8, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
                                                                                                     x_p1[:, 1, :, :],
                                                                                                     x_p2[:, 1, :, :],
                                                                                                     x_p3[:, 1, :, :]),
                                                                                                    1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 8:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
                                                                                                   x_p1[:, 2, :, :],
                                                                                                   x_p2[:, 2, :, :],
                                                                                                   x_p3[:, 2, :, :]),
                                                                                                  1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_p0, x_p1, x_p2, x_p3
        else:
            return x_fusion

def pad_tensor(tensor, target_height, target_width):
    _, _, h, w = tensor.size()
    pad_h = target_height - h
    pad_w = target_width - w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(tensor, pad, "constant", 0)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.bn(self.conv(x)))
        return x
    


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class Dense(nn.Module):  # 用来预测T的模块 （原有模块，直接在输入的基础上去预测T）
    def __init__(self):
        super(Dense, self).__init__()
        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionBlock(768, 128)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)

        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()

        # T
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = F.interpolate  # 使用 interpolate 代替 upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # 全连接层用于估计大气光值
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 3 * 256 * 256)

    def forward(self, x):
        ## 256x256
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1 = self.dense_block1(x0)
        x1 = self.trans_block1(x1)

        ### 32x32
        x2 = self.trans_block2(self.dense_block2(x1))

        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print("x4.shape", x4.shape)  # 打印x4的形状

        x42 = torch.cat([x4, x2], 1)
        x42 = self.dense_block5(x42)
        x5 = self.trans_block5(x42)

        x5 = pad_tensor(x5, x1.size(2), x1.size(3))  # 对x5进行填充以匹配x1的尺寸
        # print("x5.shape", x5.shape)  # 打印x5的形状

        x52 = torch.cat([x5, x1], 1)
        x6 = self.trans_block6(self.dense_block6(x52))

        x7 = self.trans_block7(self.dense_block7(x6))

        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = pad_tensor(x8, x.size(2), x.size(3))  # 对x8进行填充以匹配输入x的尺寸
        x8 = torch.cat([x8, x], 1)
        # print("x8.shape", x8.shape)  # [16, 19, 256, 256]

        x9 = self.relu(self.conv_refin(x8))
        # print("x9.shape", x9.shape)  # [16, 20, 256, 256]


        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        # print("x101.shape", x101.shape)  # 打印x101的形状
        x102 = F.avg_pool2d(x9, 16)
        # print("x102.shape", x102.shape)  # 打印x102的形状
        x103 = F.avg_pool2d(x9, 8)
        # print("x103.shape", x103.shape)  # 打印x103的形状
        x104 = F.avg_pool2d(x9, 4)
        # print("x104.shape", x104.shape)  # 打印x104的形状



        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # print("dehaze1.shape:" ,dehaze.shape)
        dehaze = self.tanh(self.refine3(dehaze))
        # print("dehaze(t)_shape:" ,dehaze.shape)

        return dehaze


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G2, self).__init__()
        
        # Encoder
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        self.layer2 = blockUNet(nf, nf*2, 'layer2', transposed=False, bn=True, relu=False, dropout=False)
        self.layer3 = blockUNet(nf*2, nf*4, 'layer3', transposed=False, bn=True, relu=False, dropout=False)
        self.layer4 = blockUNet(nf*4, nf*8, 'layer4', transposed=False, bn=True, relu=False, dropout=False)
        self.layer5 = blockUNet(nf*8, nf*8, 'layer5', transposed=False, bn=True, relu=False, dropout=False)
        self.layer6 = blockUNet(nf*8, nf*8, 'layer6', transposed=False, bn=True, relu=False, dropout=False)
        self.layer7 = blockUNet(nf*8, nf*8, 'layer7', transposed=False, bn=True, relu=False, dropout=False)
        self.layer8 = blockUNet(nf*8, nf*8, 'layer8', transposed=False, bn=True, relu=False, dropout=False)

        # Decoder
        self.dlayer8 = blockUNet(nf*8, nf*8, 'dlayer8', transposed=True, bn=False, relu=True, dropout=True)
        self.dlayer7 = blockUNet(nf*8*2, nf*8, 'dlayer7', transposed=True, bn=True, relu=True, dropout=True)
        self.dlayer6 = blockUNet(nf*8*2, nf*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=True)
        self.dlayer5 = blockUNet(nf*8*2, nf*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=False)
        self.dlayer4 = blockUNet(nf*8*2, nf*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=False)
        self.dlayer3 = blockUNet(nf*4*2, nf*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=False)
        self.dlayer2 = blockUNet(nf*2*2, nf, 'dlayer2', transposed=True, bn=True, relu=True, dropout=False)
        self.dlayer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, output_nc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.layer1(x) # [16, 8, 128, 128]
        print("out1:", out1.shape)
        out2 = self.layer2(out1) # [16, 16, 64, 64])

        print("out2:", out2.shape)
        out3 = self.layer3(out2) # [16, 32, 32, 32]
        print("out3:", out3.shape)
        out4 = self.layer4(out3) # [16, 64, 16, 16]
        print("out4:", out4.shape)
        out5 = self.layer5(out4) # [16, 64, 8, 8]
        print("out5:", out5.shape)
        out6 = self.layer6(out5) # [16, 64, 4, 4]
        print("out6:", out6.shape)
        out7 = self.layer7(out6) # [16, 64, 2, 2]
        print("out7:", out7.shape)
        out8 = self.layer8(out7) # [16, 64, 1, 1]
        print("out8:", out8.shape)
        dout8 = self.dlayer8(out8) # [16, 64, 2, 2]
        print("dout8:", dout8.shape)
        dout8_out7 = torch.cat([dout8, out7], 1) # [16, 128, 2, 2]
        print("dout8_out7:", dout8_out7.shape)
        dout7 = self.dlayer7(dout8_out7) # [16, 64, 4, 4]
        print("dout7:", dout7.shape)
        dout7_out6 = torch.cat([dout7, out6], 1) # [16, 128, 4, 4]
        print("dout7_out6:", dout7_out6.shape)
        dout6 = self.dlayer6(dout7_out6) # [16, 64, 8, 8]
        print("dout6:", dout6.shape)
        dout6_out5 = torch.cat([dout6, out5], 1) # [16, 128, 8, 8]
        print("dout6_out5:", dout6_out5.shape)
        dout5 = self.dlayer5(dout6_out5) # [16, 64, 16, 16]
        print("dout5:", dout5.shape)
        dout5_out4 = torch.cat([dout5, out4], 1) # [16, 128, 16, 16]
        print("dout5_out4:", dout5_out4.shape)
        dout4 = self.dlayer4(dout5_out4) # [16, 32, 32, 32]
        print("dout4:", dout4.shape)
        dout4_out3 = torch.cat([dout4, out3], 1) # [16, 64, 32, 32]
        print("dout4_out3:", dout4_out3.shape)
        dout3 = self.dlayer3(dout4_out3) # [16, 16, 64, 64]
        print("dout3:", dout3.shape)
        dout3_out2 = torch.cat([dout3, out2], 1) # [16, 32, 64, 64]
        print("dout3_out2:", dout3_out2.shape)
        dout2 = self.dlayer2(dout3_out2) # [16, 8, 128, 128]
        print("dout2:", dout2.shape)
        dout2_out1 = torch.cat([dout2, out1], 1) # [16, 16, 128, 128]
        print("dout2_out1:", dout2_out1.shape)
        dout1 = self.dlayer1(dout2_out1) # [16, 3, 256, 256]
        print("dout1:", dout1.shape)
        return dout1

# 增大对比度模块
def clahe_contrast_enhancement(image_tensor, clip_limit=1.0, grid_size=(8, 8)):
    """
    使用局部自适应直方图均衡化（CLAHE）对输入图像张量进行对比度增强。

    Args:
        image_tensor (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
        clip_limit (float, optional): CLAHE中的截断限制。默认为2.0。
        grid_size (tuple, optional): CLAHE中用于计算直方图的网格大小。默认为 (8, 8)。

    Returns:
        torch.Tensor: 对比度增强后的图像张量，与输入图像张量具有相同的形状。
    """
    # 使用detach()方法创建不需要梯度计算的副本
    image_np = image_tensor.detach().permute(0, 2, 3, 1).cpu().numpy()

    # 对每个图像应用CLAHE
    for i in range(image_np.shape[0]):
        # 将图像从 [0, 255] 范围转换为 [0, 1] 范围
        image_uint8 = (image_np[i] * 255).astype(np.uint8)

        # 转换为LAB颜色空间
        lab_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)

        # 对亮度通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

        # 转换回RGB颜色空间
        enhanced_image_uint8 = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

        # 将图像从 [0, 1] 范围转换回 [0, 255] 范围
        enhanced_image = enhanced_image_uint8.astype(np.float32) / 255.0

        # 将增强后的图像存储回原始数组中
        image_np[i] = enhanced_image

    # 将NumPy数组转换回图像张量
    enhanced_image_tensor = torch.from_numpy(image_np).permute(0, 3, 1, 2).to(image_tensor.device)

    return enhanced_image_tensor

# TODO 原始模型是支持任意尺寸输入的，如何做到这一点
# TODO 模型如果不支持任意尺寸输入，如何在测试集中也使用crop_size？或者说，如何把测试集还原回去
# TODO 看一下tools中有什么可以用的工具
class MyModel(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d'):
        super(MyModel, self).__init__()
        self.num_features = num_features

        assert arch in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=False)
        del backbone.fc
        self.backbone = backbone

        # newly added
        # 传输估计模块
        self.t = Dense()
        # 增大局部对比度模块
        # self.a = G2(input_nc=3,output_nc=3, nf=8)
        # ----same layers----
        self.down1 = nn.Sequential(nn.Conv2d(256, num_features, kernel_size=1), nn.SELU())
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()        )
        self.visualization_counter = 0  # 初始化可视化计数器

        # self.t = nn.Sequential(
        #     nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid())
        # 大气光值估计模块(卷积)
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid())

        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1))

        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        self.attention4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        # newly added 高斯滤波分解层
        # self.attention5 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        # # newly added 拉普拉斯金字塔层
        # self.attention6 = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
        #     nn.Conv2d(num_features, num_features * 4, kernel_size=1))

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1))
        
        self.j1 = nn.Sequential(nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),nn.Conv2d(num_features // 2, 3, kernel_size=1))
        self.j2 = nn.Sequential(nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),nn.Conv2d(num_features // 2, 3, kernel_size=1))
        self.j3 = nn.Sequential(nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),nn.Conv2d(num_features // 2, 3, kernel_size=1))
        self.j4 = nn.Sequential(nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),nn.Conv2d(num_features // 2, 3, kernel_size=1))
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1)
        )
        # ----end of same layers----
        # ----new layers----
        # self.pyramid_pooling = HorizontalPoolingPyramid()

        # 定义高斯滤波器
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.gaussian_filter.weight.data = torch.tensor(
            [[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]*3
        ).repeat(3, 1, 1, 1).float() / 16

        # 定义拉普拉斯金字塔层
        self.laplacian_pyramid = nn.ModuleList([nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
                                                for _ in range(4)])
        # Define refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.SELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.SELU()
        )

        # self.residual_connection = nn.Sequential(
        #     nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU()
        # )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features * 4, num_features // 8, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features // 8, num_features * 4, kernel_size=1), nn.Sigmoid()
        )
        # ----end of new layers----
        self.visualization_path = "visualization"
        os.makedirs(self.visualization_path, exist_ok=True)

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True


    # def forward(self, x0, x0_hd=None):
    #     # 预处理图像
    #     x0_wb, x0_ce, x0_gc = preprocess_image(x0)
        
    #     # 分别输入网络得到三个输出
    #     output_wb, wbx_phy, wbx_j1, wbx_j2, wbx_j3, wbx_j4, wbt, wba = self.forward_single(x0_wb)
    #     output_ce, cex_phy, cex_j1, cex_j2, cex_j3, cex_j4, cet, cea = self.forward_single(x0_ce)
    #     output_gc, gcx_phy, gcx_j1, gcx_j2, gcx_j3, gcx_j4, gct, gca = self.forward_single(x0_gc)
        
    #     # 融合三个输出
    #     output_fusion = self.fusion(output_wb, output_ce, output_gc)
        
    #     # if not self.training:
    #     #     self.visualize(x0, output_fusion)
        
    #     if self.training:
    #         x_phy = self.fusion(wbx_phy, cex_phy, gcx_phy)
    #         x_j1 = self.fusion(wbx_j1, cex_j1, gcx_j1)
    #         x_j2 = self.fusion(wbx_j2, cex_j2, gcx_j2)
    #         x_j3 = self.fusion(wbx_j3, cex_j3, gcx_j3)
    #         x_j4 = self.fusion(wbx_j4, cex_j4, gcx_j4)
    #         t = self.fusion(wbt, cet, gct)
    #         a = self.fusion(wba, cea, gca)

    #         return output_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t, a
    #     else:
    #         return output_fusion

    def forward(self, x0, x0_hd=None):
        # 在输入是增加局部对比度
        self.visualization_counter += 1  # 每次前向传播增加计数器

        # self.visualize(x0, prefix=f"{self.visualization_counter}_before_contrast_enhancement")
        
        x0 = clahe_contrast_enhancement(x0)
        # self.visualize(x0, prefix=f"{self.visualization_counter}_after_contrast_enhancement")
        # ----same layers----
        
        x = (x0 - self.mean) / self.std # [16, 3, 256, 256]

        # print("x.shape111:",x.shape)
        #TODO 能不能不直接用原图去预测t，用提取后的特征去预测t
        # 参考S21对预测T作出的修改
        t = self.t(x)
        backbone = self.backbone
        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        layer0 = backbone.maxpool(layer0)
        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)
        #TODO 多级特征提取器，能不能换

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)
        # TODO 使用转置卷积块替代上采样块
        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)
        # 绿色的那一大坨（MLF）的结果


        n, c, h, w = down1.size()
        # down1.shape: torch.Size([16, 128, 64, 64])
        # concat.shape: torch.Size([16, 512, 64, 64])

        # ----same layers----
        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 4, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :]
        f_phy = self.refine(f_phy) + f_phy # [16, 128, 64, 64]
        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
            down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1
        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
            down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2
        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
            down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3
        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
            down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std
        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))
        a = self.a(f_phy) # 对a的预测[16, 1, 1, 1]
        # t = self.ta(f_phy)
        # t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear') # 对t的预测[16, 1, 256, 256]
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.) # [16, 3, 256, 256]
        # concat.shape: torch.Size([16, 512, 64, 64])

        # 一个有用的模块
        fusion = self.dilated_conv(concat)
        fusion = self.attention_fusion(fusion)
        # print("fusion.shape:", fusion.shape)#[16, 15, 64, 64]

        attention_fusion = F.upsample(fusion, size=x0.size()[2:], mode='bilinear') # [16, 15, 256, 256]
        # 那一大坨W0~W4

        x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
                        torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                    x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
        # 3个通道分别处理 第一个通道

        x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
                        torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                    x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
        # 第二个通道

        x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
                        torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                    x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
        # 第三个通道

        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)

        # newly added 在输出时增大局部对比度
        # x_fusion = clahe_contrast_enhancement(x_fusion)

        # self.visualize(x_fusion, prefix=f"dehaze_{self.visualization_counter}")
        # ----same layers----
        
        # if not self.training:
        #     self.visualize(x0, t)
        

        if self.training:
            return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t, a.view(x.size(0), -1)
        else:
            return x_fusion
        
    def visualize(self, x, prefix=""):
        x = x.detach().cpu().numpy()
        plt.imshow(x[0].transpose(1, 2, 0))
        plt.title('Input Image')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(self.visualization_path, f'{prefix}_visualization_{timestamp}.png'))
        plt.close()

    def fusion(self, output_wb, output_ce, output_gc):
        # 融合策略，可以是加权平均、最大值选取等
        fused_output = (output_wb + output_ce + output_gc) / 3
        return fused_output
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x