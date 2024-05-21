import torch

#最大对比度假设损失,鼓励图像有更大的对比度
def contrast_loss(output, target):
    return torch.mean(torch.abs(torch.std(output, dim=[2, 3]) - torch.std(target, dim=[2, 3])))


# 色调差异损失,鼓励不同区域之间的色调变化
def tone_loss(output):
    local_var = torch.var(output, dim=[2, 3])
    return torch.mean(local_var)