# 结构相似性（SSIM）

SSIM 是一种用于比较两幅图像相似性的指标，它综合了亮度、对比度和结构三个方面的信息。SSIM 的计算基于人类视觉系统的感知机制，旨在提供更好地反映人眼对图像质量感知的指标。

## SSIM 计算过程

1. **亮度比较（Luminance Comparison）：** 计算图像的亮度平均值。亮度在 SSIM 中对应于图像的均值。
2. **对比度比较（Contrast Comparison）：** 计算图像的标准差，反映图像的对比度。对比度在 SSIM 中对应于图像的标准差。
3. **结构比较（Structure Comparison）：** 通过计算图像的互相关系数，衡量图像的结构相似度。结构在 SSIM 中对应于图像的结构信息。

## SSIM 公式

SSIM 公式的一般形式如下：

$$
\text{SSIM}(x, y) = \frac{{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}}{{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}}
$$

其中：

- $x$ 和 $y$ 是待比较的两幅图像。
- $\mu_x$、$\mu_y$ 分别是 $x$ 和 $y$ 的亮度均值。
- $\sigma_x$、$\sigma_y$ 分别是 $x$ 和 $y$ 的标准差。
- $\sigma_{xy}$ 是 $x$ 和 $y$ 之间的协方差。
- $C_1$ 和 $C_2$ 是为了稳定性而添加的常数。

SSIM 越接近 1，表示两幅图像越相似；越接近 0，表示两幅图像越不相似。

# 峰值信噪比（PSNR）

PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）是一种用于度量图像质量的指标。它通过比较图像的原始版本和经过压缩或其他处理后的版本之间的差异来评估图像的失真程度。PSNR的计算基于图像的峰值信号和均方误差之比，通常用来衡量压缩算法的性能，以及在图像处理中的信号失真程度。

PSNR的计算公式为：

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{{\text{MAX}^2}}{{\text{MSE}}} \right)
$$

其中：
- $\text{MAX}$ 是图像中像素值的最大可能值（通常为255，对于8位图像）。
- $\text{MSE}$ 是图像的均方误差（Mean Squared Error），即原始图像与处理后图像之间的差异的平方和的均值。

PSNR的值越高，表示图像的质量损失越小，失真程度越低。



# 实验结果

## 测试1：20000 epoch train_ohaze（使用ohaze数据集训练与测试）

| Image (Outdoor) | PSNR    | SSIM    | VIF     | MSE     |
|-----------------|---------|---------|---------|---------|
| 40_outdoor_hazy | 22.5420 | 0.7586  | 0.7643  | 0.0056  |
| 39_outdoor_hazy | 20.1145 | 0.7143  | 0.7728  | 0.0097  |
| 38_outdoor_hazy | 23.5469 | 0.7781  | 1.0254  | 0.0044  |
| 37_outdoor_hazy | 22.3279 | 0.6981  | 0.7238  | 0.0059  |
| 36_outdoor_hazy | 22.6933 | 0.7549  | 0.7318  | 0.0054  |

**Average:**
- L1 Loss: 0.059564
- Average PSNR: 22.244909
- Average SSIM: 0.740812
- Average VIF: 0.803607
- Average MSE: 0.006191


## 测试2：40000 epoch train（使用RESIDE数据集训练与测试）

### 在RESIDE数据测试集中测试

**Average**
- L1 Loss: 0.012660 
- Average PSNR: 34.958425
- Average SSIM: 0.974524
- Average VIF: 1.006431
- Average MSE: 0.000359

### 在ohaze数据测试集中测试

| Image (Outdoor) | PSNR    | SSIM    | VIF     | MSE     |
|-----------------|---------|---------|---------|---------|
| 40_outdoor_hazy | 17.6797 | 0.6992  | 0.7826  | 0.0171  |
| 39_outdoor_hazy | 14.7837 | 0.6493  | 0.4540  | 0.0332  |
| 38_outdoor_hazy | 14.8198 | 0.6975  | 0.5917  | 0.0330  |
| 37_outdoor_hazy | 16.3474 | 0.6416  | 0.4661  | 0.0232  |
| 36_outdoor_hazy | 18.1092 | 0.6869  | 0.5297  | 0.0155  |

**Average:**
- L1 Loss: 0.120498
- Average PSNR: 16.347973
- Average SSIM: 0.674902
- Average VIF: 0.564828
- Average MSE: 0.024381



## 可视化1：version0_ohaze测试集，不同训练集下效果对比
从左至右分别为原始带雾图片，RESIDE数据集训练后测试，OHAZE数据集训练后测试，无雾图
<img src='doc/img/ohaze_36_compare.png' height='250px'>


## 只修改预测T的_version_1

[SOTS] L1: 0.012674, PSNR: 34.955098, SSIM: 0.974479, VIF: 1.006637, MSE: 0.000360

## 修改预测T的并增加了dialtion_version_2

### HazeRD训练，SOTS测试
[SOTS] L1: 0.102853, PSNR: 17.769336, SSIM: 0.775961, VIF: 0.840989, MSE: 0.019193

## 增加了颜色损失函数和最大化对比度_version_4

[SOTS] L1: 0.080793, PSNR: 20.157936, SSIM: 0.867605, VIF: 0.955109, MSE: 0.015375

## 引入了高低频损失_version_6

## 引入了GAN_version_7
[SOTS] L1: 0.019571, PSNR: 31.257451, SSIM: 0.966829, VIF: 1.010530, MSE: 0.000840


### 可视化2：version7
在OHaze中，我们的算法1对于图像渲染错误颜色较少，但是在细节构建方面略显模糊，并且出现了网格的边界。于是我们引入了小波变换，去捕捉频域上的信息，
<img src='doc/img/ohaze_40_compare_v7.png' height='250px'>
<img src='doc/img/SOTS_1405_compare_v7.png' height='250px'>

## 重采样：version_9
| Image (Outdoor) | PSNR    | SSIM    | VIF     | MSE     |
|-----------------|---------|---------|---------|---------|
| 40_outdoor_hazy | 23.5511 | 0.7856  | 0.7760  | 0.0044  |
| 39_outdoor_hazy | 21.1983 | 0.7431  | 0.7055  | 0.0076  |
| 38_outdoor_hazy | 25.9009 | 0.7984  | 0.9419  | 0.0026  |
| 37_outdoor_hazy | 22.2040 | 0.7100  | 0.7136  | 0.0060  |
| 36_outdoor_hazy | 24.8683 | 0.7790  | 0.7813  | 0.0033  |
[O-Haze] L1: 0.052218, PSNR: 23.544525, SSIM: 0.763222, VIF: 0.783652, MSE: 0.004771
<img src='doc/img/ohaze_37_compare_v9.png' width='750px'>
[SOTS] L1: 0.018361, PSNR: 31.582130, SSIM: 0.977500, VIF: 0.999038, MSE: 0.000802
[HazeRD] L1: 0.164127, PSNR: 15.751373, SSIM: 0.837204, VIF: 0.734832, MSE: 0.054203, CIEDE2000: 14.546041



## version_10 OHaze网络中加了膨胀卷积，并且把训练集换成40个图

predicting for O-Haze (1/10) [40_outdoor_hazy]: PSNR 27.9137, SSIM 0.8140, VIF 0.9000, MSE 0.0016, CIEDE 3.4875
predicting for O-Haze (2/10) [39_outdoor_hazy]: PSNR 25.1250, SSIM 0.7682, VIF 0.8362, MSE 0.0031, CIEDE 6.2746
predicting for O-Haze (3/10) [38_outdoor_hazy]: PSNR 27.5630, SSIM 0.8054, VIF 0.8572, MSE 0.0018, CIEDE 4.2267
predicting for O-Haze (4/10) [37_outdoor_hazy]: PSNR 24.2018, SSIM 0.7245, VIF 0.7470, MSE 0.0038, CIEDE 5.3036
predicting for O-Haze (5/10) [36_outdoor_hazy]: PSNR 27.8243, SSIM 0.7948, VIF 0.8631, MSE 0.0017, CIEDE 3.9966
predicting for O-Haze (6/10) [41_outdoor_hazy]: PSNR 26.2791, SSIM 0.8546, VIF 1.0271, MSE 0.0024, CIEDE 4.1429
predicting for O-Haze (7/10) [42_outdoor_hazy]: PSNR 24.4594, SSIM 0.7245, VIF 0.8761, MSE 0.0036, CIEDE 5.1982
predicting for O-Haze (8/10) [43_outdoor_hazy]: PSNR 28.4866, SSIM 0.8343, VIF 0.9813, MSE 0.0014, CIEDE 3.8141
predicting for O-Haze (9/10) [44_outdoor_hazy]: PSNR 25.3821, SSIM 0.8042, VIF 0.8286, MSE 0.0029, CIEDE 4.6711
predicting for O-Haze (10/10) [45_outdoor_hazy]: PSNR 23.8995, SSIM 0.7788, VIF 0.7826, MSE 0.0041, CIEDE 6.7396
[O-Haze] L1: 0.037661, PSNR: 26.113449, SSIM: 0.790316, VIF: 0.869915, MSE: 0.002622, CIEDE2000: 4.785487


## version_10 RESIDE网络中修改了颜色损失的权值，还没来得及训练

TODO:重新训一下baseline的DM2F和DM2F_woPhy，为了填表格
重新训MyModel，效果太差了，想想怎么把奇怪的颜色斑块和网格给去掉
消融实验

baseline1
[O-Haze] L1: 0.035569, PSNR: 25.645535, SSIM: 0.777491, VIF: 0.892133, MSE: 0.002391, CIEDE2000: 4.284915

baseline2
[HazeRD] L1: 0.207146, PSNR: 13.833413, SSIM: 0.828152, VIF: 0.554123, MSE: 0.080635, CIEDE2000: 17.294823



alg2
[HazeRD] L1: 0.150652, PSNR: 16.161701, SSIM: 0.828488, VIF: 0.899092, MSE: 0.047187, CIEDE2000: 13.265078
