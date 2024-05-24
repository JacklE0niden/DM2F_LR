# newly added
# 1.修改了backbone
# 2.对于图像的模糊性特征进行精细化提取，加入了RefinementNet
# 3.加入了颜色一致性损失
# 4.加入了拉普拉斯多尺度高低频损失函数
# TODO 加入小波变换的多尺度损失
# TODO 加入膨胀卷积
# TODO 加入对于t的预测
# TODO 加入后处理的网络
class DM2FNet_woPhy_My(Base_OHAZE):
    def __init__(self, use_refine=False, use_sep=False, use_final=False, num_features=64, arch='efficientnet_b0'):
        super(DM2FNet_woPhy_My, self).__init__()
        self.num_features = num_features
        self.use_refine = use_refine
        self.use_sep = use_sep
        self.use_final = use_final
        self.num_features = num_features

        # NOTE 修改了backbone结构
        assert arch in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                        'densenet121', 'densenet169', 'densenet201', 'mobilenet_v2', 'efficientnet_b0']
        self.arch = arch

        if 'resnet' in arch or 'resnext' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.fc
            self.backbone = backbone
        elif 'densenet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features
        elif 'mobilenet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features
        elif 'efficientnet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features

        if 'resnet' in arch or 'resnext' in arch:
            down_channels = [256, 512, 1024, 2048]
        elif 'densenet' in arch:
            down_channels = [64, 128, 256, 512]
        elif 'mobilenet' in arch:
            down_channels = [24, 40, 80, 112, 320]  # EfficientNet specific down_channels, adjust if necessary
        elif 'efficientnet' in arch:
            down_channels = [16, 24, 40, 80]
        # print("backbone", self.backbone)
        # backbone ResNet

        # NOTE 修改，加入了精细化网络，对于x_p0的图像模糊特征进行精细化提取
        if self.use_refine:
            self.refine_net = RefinementNet()
        
        if self.use_sep:
            self.refine_net1 = RefinementNet()
            self.refine_net2 = RefinementNet()
            self.refine_net3 = RefinementNet()
        
        if self.use_final:
            self.refine_net_final = RefinementNet()

        self.down0 = nn.Sequential(
            nn.Conv2d(32, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(down_channels[0], num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(down_channels[1], num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(down_channels[2], num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(down_channels[3], num_features, kernel_size=1), nn.SELU()
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

        # layer0 = backbone.conv1(x)
        # layer0 = backbone.bn1(layer0)
        # layer0 = backbone.relu(layer0)
        # layer0 = backbone.maxpool(layer0)

        # layer1 = backbone.layer1(layer0)
        # layer2 = backbone.layer2(layer1)
        # layer3 = backbone.layer3(layer2)
        # layer4 = backbone.layer4(layer3)
        
        # Forward through the EfficientNet backbone
        layer_outputs = []
        x_ = x
        for layer in backbone:
            x_ = layer(x_)
            layer_outputs.append(x_)

        layer0 = layer_outputs[0]
        # print("layer0.shape:", layer0.shape)
        layer1 = layer_outputs[1]
        # print("layer1.shape:", layer1.shape)
        layer2 = layer_outputs[2]
        # print("layer2.shape:", layer2.shape)
        layer3 = layer_outputs[3]
        # print("layer3.shape:", layer3.shape)
        layer4 = layer_outputs[4]
        # print("layer4.shape:", layer4.shape)
        


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

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        x_p0 = torch.exp(log_x0 + F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0, max=1)
        if self.use_refine:
            x_p0 = self.refine_net(x_p0, x)

        x_p1 = ((x + F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)\
            .clamp(min=0., max=1.)
        if self.use_refine:
            if self.use_sep:
                x_p1 = self.refine_net1(x_p1, x)
            else:
                x_p1 = self.refine_net(x_p1, x)

        log_x_p2_0 = torch.log(
            ((x + F.upsample(self.p2_0(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)
                .clamp(min=1e-8))
        x_p2 = torch.exp(log_x_p2_0 + F.upsample(self.p2_1(f), size=x0.size()[2:], mode='bilinear'))\
            .clamp(min=0., max=1.)
        if self.use_refine:
            if self.use_sep:
                x_p2 = self.refine_net2(x_p2, x)
            else:
                x_p2 = self.refine_net(x_p2, x)

        log_x_p3_0 = torch.exp(log_log_x0_inverse + F.upsample(self.p3_0(f), size=x0.size()[2:], mode='bilinear'))
        x_p3 = torch.exp(-log_x_p3_0 + F.upsample(self.p3_1(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0,
                                                                                                            max=1)
        if self.use_refine:
            if self.use_sep:
                x_p3 = self.refine_net3(x_p3, x)
            else:
                x_p3 = self.refine_net(x_p3, x)
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





class MyModel(Base):
    def __init__(self, num_features=128, arch='efficientnet_b0'):
        super(MyModel, self).__init__()
        self.num_features = num_features

        assert arch in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                        'densenet121', 'densenet169', 'densenet201', 'mobilenet_v2', 'efficientnet_b0']
        self.arch = arch

        if 'resnet' in arch or 'resnext' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.fc
            self.backbone = backbone
        elif 'densenet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features
        elif 'mobilenet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features
        elif 'efficientnet' in arch:
            backbone = models.__dict__[arch](pretrained=True)
            del backbone.classifier
            self.backbone = backbone.features

        if 'resnet' in arch or 'resnext' in arch:
            down_channels = [256, 512, 1024, 2048]
        elif 'densenet' in arch:
            down_channels = [64, 128, 256, 512]
        elif 'mobilenet' in arch:
            down_channels = [24, 40, 80, 112, 320]  # EfficientNet specific down_channels, adjust if necessary
        elif 'efficientnet' in arch:
            down_channels = [16, 24, 40, 80]

        self.down1 = nn.Sequential(
            nn.Conv2d(down_channels[0], num_features, kernel_size=1), nn.SELU(),
            PALayer(num_features),
            CALayer(num_features)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(down_channels[1], num_features, kernel_size=1), nn.SELU(),
            PALayer(num_features),
            CALayer(num_features)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(down_channels[2], num_features, kernel_size=1), nn.SELU(),
            PALayer(num_features),
            CALayer(num_features)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(down_channels[3], num_features, kernel_size=1), nn.SELU(),
            PALayer(num_features),
            CALayer(num_features)
        )
        #newly added
        # 上采样块
        # self.d_upsample1 = UpsampleBlock(128, 64)  # 输入通道数为128，输出通道数为64
        # self.d_upsample2 = UpsampleBlock(128, 64)
        # self.d_upsample3 = UpsampleBlock(128, 64)

        # newly added
        # 传输估计模块
        # self.t = Dense()
        # 增大局部对比度模块
        # self.a = G2(input_nc=3,output_nc=3, nf=8)
        # # ----same layers----
        # self.down1 = nn.Sequential(nn.Conv2d(256, num_features, kernel_size=1), nn.SELU())
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()        )
        # self.down3 = nn.Sequential(
        #     nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()        )
        # self.down4 = nn.Sequential(
        #     nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()        )
        self.visualization_counter = 0  # 初始化可视化计数器
        
        # newly added P02 
        # FA模块分别在通道方式和像素方式特征中组合通道注意力和像素注意力
        # 输入与输出形状相同
        # self.ca_layer1 = CALayer(num_features)
        # self.pa_layer1 = PALayer(num_features)
        
        
        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid())
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
        self.attention_fusion = nn.Sequential( # 最右边那个权重
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1)
        )
        # ----end of same layers----
        # ----new layers----
        # self.pyramid_pooling = HorizontalPoolingPyramid()

        # # 定义高斯滤波器
        # self.gaussian_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        # self.gaussian_filter.weight.data = torch.tensor(
        #     [[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]*3
        # ).repeat(3, 1, 1, 1).float() / 16

        # 定义拉普拉斯金字塔层
        # self.laplacian_pyramid = nn.ModuleList([nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        #                                         for _ in range(4)])
        # Define refinement layers

        self.refine_net = RefinementNet()
        # self.refine_net1 = RefinementNet()
        # self.refine_net2 = RefinementNet()
        # self.refine_net3 = RefinementNet()
        # self.refine_net4 = RefinementNet()

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



    def forward(self, x0, x0_hd=None):
        # NOTE 方法1：在输入时增加局部对比度
        # newly added
        # self.visualization_counter += 1  # 每次前向传播增加计数器
        # self.visualize(x0, prefix=f"{self.visualization_counter}_before_contrast_enhancement")
        # x0 = clahe_contrast_enhancement(x0)
        # self.visualize(x0, prefix=f"{self.visualization_counter}_after_contrast_enhancement")
        # ----same layers----
        x = (x0 - self.mean) / self.std # [16, 3, 256, 256]
        # print("x.shape111:",x.shape)
        #TODO 能不能不直接用原图去预测t，用提取后的特征去预测t
        # 参考S21对预测T作出的修改
        # t = self.t(x)
        # #TODO 多级特征提取器，能不能换
        # layer0.shape: torch.Size([16, 64, 64, 64])
        # layer1.shape: torch.Size([16, 256, 64, 64])
        # layer2.shape: torch.Size([16, 512, 32, 32])
        # layer3.shape: torch.Size([16, 1024, 16, 16])
        # layer4.shape: torch.Size([16, 2048, 8, 8])

        # newly added-----------------
        backbone = self.backbone

        layer_outputs = []
        x_ = x
        for layer in backbone:
            x_ = layer(x_)
            layer_outputs.append(x_)

        layer0 = layer_outputs[0]
        layer1 = layer_outputs[1]
        layer2 = layer_outputs[2]
        layer3 = layer_outputs[3]
        layer4 = layer_outputs[4]
        
        # layer0.shape: torch.Size([16, 32, 128, 128])
        # layer1.shape: torch.Size([16, 16, 128, 128])
        # layer2.shape: torch.Size([16, 24, 64, 64])
        # layer3.shape: torch.Size([16, 40, 32, 32])
        # layer4.shape: torch.Size([16, 80, 16, 16])
        # down1.shape: torch.Size([16, 128, 64, 64])


        # newly added end-----------------

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)
        # TODO 使用转置卷积块替代上采样块
        # print("down1.shape", down1.shape)
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

        # newly added P02
        # f_phy = self.ca_layer1(f_phy) # 经过AFIM之后的特征

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
            down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1
        # newly added
        # f1 = self.pa_layer1(f1)

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
            down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2
        # newly added
        # f2 = self.pa_layer1(f2)

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
            down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3
        # newly added
        # f3 = self.pa_layer1(f3)

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
            down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4
        # newly added
        # f4 = self.pa_layer1(f4)

        

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std
        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))
        a = self.a(f_phy) # 对a的预测[16, 1, 1, 1]
        # NOTE:modified
        # t = self.ta(f_phy)
        # 对t和a的预测是经过AFIM之后才进行的，并且只用了两个卷积层，似乎有一些草率，能否修改呢？
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear') # 对t的预测[16, 1, 256, 256] 
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # newly added 对特征进行细化
        x_j1 = self.refine_net(x_j1)
        
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        # print("x.shape", x.shape)
        # print("r2.shape:", r2.shape)
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)
        # newly added 对特征进行细化
        x_j2 = self.refine_net(x_j2)
        
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)
        
        # newly added 对特征进行细化
        x_j3 = self.refine_net(x_j3)


        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.) # [16, 3, 256, 256]

        # newly added 对特征进行细化
        x_j4 = self.refine_net(x_j4)
        # concat.shape: torch.Size([16, 512, 64, 64])
        
        # newly added
        # 一个有用的模块
        # 膨胀卷积，可以使得注意力机制的混合更加均匀
        fusion = self.dilated_conv(concat)
        fusion = self.attention_fusion(fusion)

        # fusion = self.attention_fusion(concat)
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