import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tools.change_image import rgb_to_lab, rgb_to_hsv

#最大对比度假设损失,鼓励图像有更大的对比度
# NOTE unused
def contrast_loss(output, target):
    return torch.mean(torch.abs(torch.std(output, dim=[2, 3]) - torch.std(target, dim=[2, 3])))


# 色调差异损失,鼓励不同区域之间的色调变化
# NOTE unused
def tone_loss(output):
    local_var = torch.var(output, dim=[2, 3])
    return torch.mean(local_var)


class ColorConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(ColorConsistencyLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        # 转换图像到LAB颜色空间
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)
        
        # 计算LAB空间的L2损失
        loss = F.mse_loss(pred_lab, target_lab)
        return self.weight * loss

    # def rgb_to_lab(self, image_tensor):
    #     # 转换为numpy数组
    #     image_np = image_tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
    #     lab_images = []
    #     for img in image_np:
    #         img_uint8 = (img * 255).astype(np.uint8)
    #         lab_image = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    #         lab_image = lab_image.astype(np.float32) / 255.0
    #         lab_images.append(lab_image)
    #     lab_images_np = np.stack(lab_images)
    #     lab_images_tensor = torch.from_numpy(lab_images_np).permute(0, 3, 1, 2).to(image_tensor.device)
    #     return lab_images_tensor
    
class LaplacianFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = torch.cat([kernel, kernel, kernel], 0)
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)
        if not self.filter.weight.is_cuda and torch.cuda.is_available():
            # 初始化权重并将其移到 GPU 上
            self.filter.weight = nn.Parameter(self.filter.weight.cuda(), requires_grad=False)
        
    def forward(self, image):
        return self.filter(image)


# S145
# DWT小波变换，变为高频和低频分量

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dtype = x.dtype  # 获取输入的原始dtype
        # print(f"Input dtype: {dtype}")
        # print(f"Before conversion - conv1x1_low dtype: {self.conv1x1_low.weight.dtype}, conv1x1_high dtype: {self.conv1x1_high.weight.dtype}")
        
        self.conv1x1_low = self.conv1x1_low.to(dtype)  # 将conv1x1_low转换为相同的dtype
        self.conv1x1_high = self.conv1x1_high.to(dtype)  # 将conv1x1_high转换为相同的dtype
        
        # print(f"After conversion - conv1x1_low dtype: {self.conv1x1_low.weight.dtype}, conv1x1_high dtype: {self.conv1x1_high.weight.dtype}")
        
        x = x.to(torch.float32)  # 将输入转换为float32
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        
        # print(f"typedwt_low_frequency: {dwt_low_frequency.dtype}, dwt_high_frequency: {dwt_high_frequency.dtype}")
        
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        
        return dwt_low_frequency.to(dtype), dwt_high_frequency.to(dtype)  # 返回结果并转换回原始dtype


def compute_multiscale_hf_lf_loss_dwt(ground_truth, predicted_output, loss_function, dwt_transform, scales=[1, 0.5, 0.25]):
    total_loss = 0
    for scale in scales:
        # 调整图像到不同的尺度
        gt_scaled = F.interpolate(ground_truth, scale_factor=scale, mode='bilinear', align_corners=True)
        pred_scaled = F.interpolate(predicted_output, scale_factor=scale, mode='bilinear', align_corners=True)
        
        # 计算残差
        residual = gt_scaled - pred_scaled
        
        # 计算DWT的低频和高频成分
        gt_low_freq, gt_high_freq = dwt_transform(gt_scaled)
        pred_low_freq, pred_high_freq = dwt_transform(pred_scaled)
        
        # 计算低频和高频成分的损失，并累加到总损失中
        total_loss += loss_function(pred_low_freq, gt_low_freq) + loss_function(pred_high_freq, gt_high_freq)
    
    # 返回平均损失
    return total_loss / len(scales)

def compute_multiscale_hf_lf_loss_lp(ground_truth, predicted_output, loss_function, laplacian_filter, scales=[1, 0.5, 0.25]):
    total_loss = 0
    for scale in scales:
        # 调整图像到不同的尺度
        gt_scaled = F.interpolate(ground_truth, scale_factor=scale, mode='bilinear', align_corners=True)
        pred_scaled = F.interpolate(predicted_output, scale_factor=scale, mode='bilinear', align_corners=True)
        
        # 计算高频和低频成分
        high_freq_pred = laplacian_filter(pred_scaled)
        low_freq_pred = pred_scaled - laplacian_filter(pred_scaled)
        high_freq_gt = laplacian_filter(gt_scaled)
        low_freq_gt = gt_scaled - laplacian_filter(gt_scaled)
        
        # 计算高频和低频成分的损失，并累加到总损失中
        total_loss += loss_function(high_freq_pred, high_freq_gt) + loss_function(low_freq_pred, low_freq_gt)
    
    # 返回平均损失
    return total_loss / len(scales)