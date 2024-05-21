import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


#最大对比度假设损失,鼓励图像有更大的对比度
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
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # 计算LAB空间的L2损失
        loss = F.mse_loss(pred_lab, target_lab)
        return self.weight * loss

    def rgb_to_lab(self, image_tensor):
        # 转换为numpy数组
        image_np = image_tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
        lab_images = []
        for img in image_np:
            img_uint8 = (img * 255).astype(np.uint8)
            lab_image = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            lab_image = lab_image.astype(np.float32) / 255.0
            lab_images.append(lab_image)
        lab_images_np = np.stack(lab_images)
        lab_images_tensor = torch.from_numpy(lab_images_np).permute(0, 3, 1, 2).to(image_tensor.device)
        return lab_images_tensor