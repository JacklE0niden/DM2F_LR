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




def ciede2000_color_diff(gt, r, KLCH=None):
    """
    Computes the CIEDE2000 color difference between two RGB color images.

    Parameters:
    rgbstd : ndarray
        Standard RGB image.
    rgbsample : ndarray
        Sample RGB image.
    KLCH : tuple, optional
        Parameters for adjusting lightness, chroma, and hue.

    Returns:
    dE00 : float
        The CIEDE2000 color difference between the two images.
    """
    # Convert RGB images to Lab color space
    labstd = cv2.cvtColor(gt, cv2.COLOR_BGR2Lab)
    labsample = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    
    # Compute Lab color difference
    dE00 = compute_ciede2000(labstd, labsample, KLCH)
    
    return dE00

def compute_ciede2000(Labstd, Labsample, KLCH=None):
    """
    Compute the CIEDE2000 color difference between two Lab color images.

    Parameters:
    Labstd : ndarray
        Standard Lab color image.
    Labsample : ndarray
        Sample Lab color image.
    KLCH : tuple, optional
        Parameters for adjusting lightness, chroma, and hue.

    Returns:
    dE00 : float
        The CIEDE2000 color difference between the two images.
    """
    h = Labstd.shape[0]
    w = Labstd.shape[1]
    c = Labstd.shape[2]
    
    Labstd = Labstd.reshape(h*w, c)
    Labsample = Labsample.reshape(h*w, c)
    
    # Ensure inputs are numpy arrays
    Labstd = np.array(Labstd)
    Labsample = np.array(Labsample)
    
    # Check input dimensions
    if Labstd.shape != Labsample.shape:
        raise ValueError('Standard and Sample sizes do not match')
    if Labstd.shape[1] != 3:
        print("Labstd.shape:", Labstd.shape)
        raise ValueError('Standard and Sample Lab vectors should be Kx3 vectors')
    
    # Set default parametric factors if not provided
    if KLCH is None:
        kl, kc, kh = 1, 1, 1
    else:
        if len(KLCH) != 3:
            raise ValueError('KLCH must be a 1x3 vector')
        kl, kc, kh = KLCH
    
    # Extract Lab channels
    Lstd, astd, bstd = Labstd[:, 0], Labstd[:, 1], Labstd[:, 2]
    Lsample, asample, bsample = Labsample[:, 0], Labsample[:, 1], Labsample[:, 2]
    
    # Compute color differences
    dE00 = compute_delta_e(Lstd, astd, bstd, Lsample, asample, bsample, kl, kc, kh)
    
    return dE00

def compute_delta_e(Lstd, astd, bstd, Lsample, asample, bsample, kl, kc, kh):
    """
    Compute the CIEDE2000 color difference between two sets of Lab color values.

    Parameters:
    Lstd, astd, bstd : array_like
        Standard Lab color values.
    Lsample, asample, bsample : array_like
        Sample Lab color values.
    kl, kc, kh : float
        Parameters for adjusting lightness, chroma, and hue.

    Returns:
    dE00 : float
        The CIEDE2000 color difference between the two Lab colors.
    """
    # Convert Lab values to numpy arrays
    # Lstd_arr, astd_arr, bstd_arr = np.array(Lstd), np.array(astd), np.array(bstd)
    # Lsample_arr, asample_arr, bsample_arr = np.array(Lsample), np.array(asample), np.array(bsample)
    
    # # Calculate color differences
    # dL = Lsample_arr - Lstd_arr
    # da = asample_arr - astd_arr
    # db = bsample_arr - bstd_arr
    
    # Calculate intermediate parameters
    C1 = np.sqrt(astd**2 + bstd**2)
    C2 = np.sqrt(asample**2 + bsample**2)
    Cabarithmean = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt((Cabarithmean**7) / (Cabarithmean**7 + 25**7)))
    
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpstd = np.sqrt(apstd**2 + bstd**2)
    Cpsample = np.sqrt(apsample**2 + bsample**2)
    
    Cpprod = Cpsample * Cpstd
    zcidx = np.where(Cpprod == 0)[0]
    
    hpstd = np.arctan2(bstd, apstd)
    hpstd = np.mod(hpstd + 2 * np.pi * (hpstd < 0), 2 * np.pi)
    hpstd[(np.abs(apstd) + np.abs(bstd)) == 0] = 0
    
    hpsample = np.arctan2(bsample, apsample)
    hpsample = np.mod(hpsample + 2 * np.pi * (hpsample < 0), 2 * np.pi)
    hpsample[(np.abs(apsample) + np.abs(bsample)) == 0] = 0
    
    dL = Lsample - Lstd
    dC = Cpsample - Cpstd
    
    dhp = hpsample - hpstd
    dhp[dhp > np.pi] -= 2 * np.pi
    dhp[dhp < -np.pi] += 2 * np.pi
    dhp[zcidx] = 0
    
    ΔH_ = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    ΔH__bar = np.abs(hpstd - hpsample)
    ΔH__bar[np.abs(hpstd - hpsample) > np.pi] -= 2 * np.pi
    ΔH__bar = np.abs(ΔH__bar)
    
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    
    hp = (hpstd + hpsample) / 2
    hp[ΔH__bar > np.pi] -= np.pi
    
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + 0.32 * np.cos(3 * hp + np.pi / 30) - 0.2 * np.cos(4 * hp - 63 * np.pi / 180)
    Sl = 1 + 0.015 * (Lp - 50)**2 / np.sqrt(20 + (Lp - 50)**2)
    Sc = 1 + 0.045 * Cp
    Sh = 1 + 0.015 * Cp * T
    
    delthetarad = (30 * np.pi / 180) * np.exp(-((180 / np.pi * hp - 275) / 25)**2)
    Rc = 2 * np.sqrt((Cp**7) / (Cp**7 + 25**7))
    RT = -np.sin(2 * delthetarad) * Rc
    
    dE00 = np.sqrt((dL / (kl * Sl))**2 + (dC / (kc * Sc))**2 + (ΔH_ / (kh * Sh))**2 + RT * (dC / (kc * Sc)) * (ΔH_ / (kh * Sh)))
    
    return dE00