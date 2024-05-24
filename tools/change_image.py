import colorsys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def rgb_to_lab(image_tensor):
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

def rgb_to_hsv(image_tensor):
    # Convert image tensor to numpy array
    image_np = image_tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
    hsv_images = []
    for img in image_np:
        img_uint8 = (img * 255).astype(np.uint8)
        hsv_image = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        hsv_image = hsv_image.astype(np.float32) / 255.0
        hsv_images.append(hsv_image)
    hsv_images_np = np.stack(hsv_images)
    hsv_images_tensor = torch.from_numpy(hsv_images_np).permute(0, 3, 1, 2).to(image_tensor.device)
    return hsv_images_tensor