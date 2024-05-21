
# def main():
#     net = DM2FNet().cuda().train()
#     # net = MyModel().cuda().train()
#     discriminator = Discriminator().cuda().train()
#     # net = nn.DataParallel(net)

#     optimizer = optim.Adam([
#         {'params': [param for name, param in net.named_parameters()
#                     if name[-4:] == 'bias' and param.requires_grad],
#          'lr': 2 * cfgs['lr']},
#         {'params': [param for name, param in net.named_parameters()
#                     if name[-4:] != 'bias' and param.requires_grad],
#          'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
#     ])
#     # discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=cfgs['lr_discriminator'])

#     if len(cfgs['snapshot']) > 0:
#         print('training resumes from \'%s\'' % cfgs['snapshot'])
#         net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
#                                                     args.exp_name, cfgs['snapshot'] + '.pth')))
#         optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
#                                                           args.exp_name, cfgs['snapshot'] + '_optim.pth')))
#         optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
#         optimizer.param_groups[1]['lr'] = cfgs['lr']

#     check_mkdir(args.ckpt_path)
#     check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
#     open(log_path, 'w').write(str(cfgs) + '\n\n')

#     train(net, optimizer, discriminator)


# def train(net, optimizer, discriminator):
#     curr_iter = cfgs['last_iter']

#     while curr_iter <= cfgs['iter_num']:
#         train_loss_record = AvgMeter()
#         loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
#         loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
#         loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
#         loss_t_record, loss_a_record = AvgMeter(), AvgMeter()

#         for data in train_loader:
#             optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
#                                               ** cfgs['lr_decay']
#             optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
#                                               ** cfgs['lr_decay']

#             haze, gt_trans_map, gt_ato, gt, _ = data
#             # haze, gt, _ = data
#             batch_size = haze.size(0)

#             haze = haze.cuda()
#             gt_trans_map = gt_trans_map.cuda()
#             gt_ato = gt_ato.cuda()
#             gt = gt.cuda()

#             optimizer.zero_grad()

#             x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)

#             loss_x_jf = criterion(x_jf, gt)
#             loss_x_j0 = criterion(x_j0, gt)
#             loss_x_j1 = criterion(x_j1, gt)
#             loss_x_j2 = criterion(x_j2, gt)
#             loss_x_j3 = criterion(x_j3, gt)
#             loss_x_j4 = criterion(x_j4, gt)

#             loss_t = criterion(t, gt_trans_map)
#             loss_a = criterion(a, gt_ato)

#             # fake_images = x_jf

#             # real_output = discriminator(gt)
#             # fake_output = discriminator(fake_images.detach())
#             # d_loss_real = criterion(real_output, torch.ones_like(real_output))
#             # d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
#             # d_loss = d_loss_real + d_loss_fake

#             # Optimize Discriminator
#             # discriminator_optimizer.zero_grad()
#             # d_loss.backward()
#             # discriminator_optimizer.step()

#             # Optimize Generator
#             generator_loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 \
#                             + 10 * loss_t + loss_a
#             # generator_loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4
#             # generator_loss.backward()
#             optimizer.step()

#             # Update recorder
#             train_loss_record.update(generator_loss.item(), batch_size)
#             loss_x_jf_record.update(loss_x_jf.item(), batch_size)
#             loss_x_j0_record.update(loss_x_j0.item(), batch_size)
#             loss_x_j1_record.update(loss_x_j1.item(), batch_size)
#             loss_x_j2_record.update(loss_x_j2.item(), batch_size)
#             loss_x_j3_record.update(loss_x_j3.item(), batch_size)
#             loss_x_j4_record.update(loss_x_j4.item(), batch_size)
#             loss_t_record.update(loss_t.item(), batch_size)
#             loss_a_record.update(loss_a.item(), batch_size)

#             curr_iter += 1

#             log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
#                   '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
#                   '[lr %.13f]' % \
#                   (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
#                    loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
#                    loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
#             print(log)
#             open(log_path, 'a').write(log + '\n')

#             if (curr_iter + 1) % cfgs['val_freq'] == 0:
#                 validate(net, curr_iter, optimizer)

#             if curr_iter > cfgs['iter_num']:
#                 break



# def validate(net, curr_iter, optimizer):
#     print('validating...')
#     net.eval()

#     loss_record = AvgMeter()

#     with torch.no_grad():
#         for data in tqdm(val_loader):
#             haze, gt, _ = data

#             haze = haze.cuda()
#             gt = gt.cuda()
#             print("haze.shape:", haze.shape)
#             dehaze = net(haze)
#             # dehaze = sliding_forward(net, haze).detach()
#             loss = criterion(dehaze, gt)
#             loss_record.update(loss.item(), haze.size(0))

#     snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
#     print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
#     torch.save(net.state_dict(),
#                os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
#     torch.save(optimizer.state_dict(),
#                os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

#     net.train()


# if __name__ == '__main__':
#     args = parse_args()
#     print("args parsed.......")
#     # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
#     cudnn.benchmark = True
#     print("gpus = ", args.gpus)
    
#     cudnn.benchmark = True
#     torch.cuda.set_device(int(args.gpus))

#     train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
#     # train_dataset = HazeRDDataset(HAZERD_ROOT, 'train', True, cfgs['crop_size'])
#     train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
#                               shuffle=True, drop_last=True)

#     val_dataset = SotsDataset(TEST_SOTS_ROOT)
#     val_loader = DataLoader(val_dataset, batch_size=8)

#     criterion = nn.L1Loss().cuda()
#     log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

#     main()

# class G2(nn.Module): # 用来预测A的模块
#   def __init__(self, input_nc, output_nc, nf):
#     super(G2, self).__init__()
#     # input is 256 x 256
#     layer_idx = 1
#     name = 'layer%d' % layer_idx
#     layer1 = nn.Sequential()
#     layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
#     # input is 128 x 128
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 64 x 64
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 32
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 16
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 8
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 4
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
#     # input is 2 x  2
#     layer_idx += 1
#     name = 'layer%d' % layer_idx
#     layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

#     ## NOTE: decoder
#     # input is 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*8
#     dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

#     #import pdb; pdb.set_trace()
#     # input is 2
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*8*2
#     dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
#     # input is 4
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*8*2
#     dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
#     # input is 8
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*8*2
#     dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
#     # input is 16
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*8*2
#     dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
#     # input is 32
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*4*2
#     dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
#     # input is 64
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     d_inc = nf*2*2
#     dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
#     # input is 128
#     layer_idx -= 1
#     name = 'dlayer%d' % layer_idx
#     dlayer1 = nn.Sequential()
#     d_inc = nf*2
#     dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
#     dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
#     dlayer1.add_module('%s_tanh' % name, nn.LeakyReLU(0.2, inplace=True))

#     self.layer1 = layer1
#     self.layer2 = layer2
#     self.layer3 = layer3
#     self.layer4 = layer4
#     self.layer5 = layer5
#     self.layer6 = layer6
#     self.layer7 = layer7
#     self.layer8 = layer8
#     self.dlayer8 = dlayer8
#     self.dlayer7 = dlayer7
#     self.dlayer6 = dlayer6
#     self.dlayer5 = dlayer5
#     self.dlayer4 = dlayer4
#     self.dlayer3 = dlayer3
#     self.dlayer2 = dlayer2
#     self.dlayer1 = dlayer1

#   def forward(self, x):
#     print("input:", x.shape)
#     out1 = self.layer1(x)
#     out2 = self.layer2(out1)
#     out3 = self.layer3(out2)
#     out4 = self.layer4(out3)
#     out5 = self.layer5(out4)
#     out6 = self.layer6(out5)
#     out7 = self.layer7(out6)
#     out8 = self.layer8(out7)
#     dout8 = self.dlayer8(out8)
#     dout8_out7 = torch.cat([dout8, out7], 1)
#     dout7 = self.dlayer7(dout8_out7)
#     dout7_out6 = torch.cat([dout7, out6], 1)
#     dout6 = self.dlayer6(dout7_out6)
#     dout6_out5 = torch.cat([dout6, out5], 1)
#     dout5 = self.dlayer5(dout6_out5)
#     dout5_out4 = torch.cat([dout5, out4], 1)
#     dout4 = self.dlayer4(dout5_out4)
#     dout4_out3 = torch.cat([dout4, out3], 1)
#     dout3 = self.dlayer3(dout4_out3)
#     dout3_out2 = torch.cat([dout3, out2], 1)
#     dout2 = self.dlayer2(dout3_out2)
#     dout2_out1 = torch.cat([dout2, out1], 1)
#     dout1 = self.dlayer1(dout2_out1)
#     return dout1

# class BottleneckBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super(BottleneckBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(4 * growth_rate)
#         self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         return torch.cat([x, out], 1)

# class TransitionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(TransitionBlock, self).__init__()
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
#     def forward(self, x):
#         out = self.conv(F.relu(self.bn(x)))
#         out = self.pool(out)
#         return out

# class Dense(nn.Module):  # 用来预测T的模块 废弃了
#     def __init__(self):
#         super(Dense, self).__init__()
        
#         ############# Block1-down 32-32  ##############
#         self.dense_block1 = BottleneckBlock(128, 64)
#         self.trans_block1 = TransitionBlock(192, 64)

#         ############# Block2-down 16-16  ##############
#         self.dense_block2 = BottleneckBlock(64, 32)
#         self.trans_block2 = TransitionBlock(96, 32)

#         ############# Block3-down  8-8  ##############
#         self.dense_block3 = BottleneckBlock(32, 16)
#         self.trans_block3 = TransitionBlock(48, 16)

#         ############# Block4-up  16-16  ##############
#         self.dense_block4 = BottleneckBlock(16, 16)
#         self.trans_block4 = TransitionBlock(32, 32)

#         ############# Block5-up  32-32 ##############
#         self.dense_block5 = BottleneckBlock(64, 32)  # Outputs 96 channels
#         self.trans_block5 = TransitionBlock(96, 64)

#         ############# Block6-up 64-64   ##############
#         self.dense_block6 = BottleneckBlock(128, 64)
#         self.trans_block6 = TransitionBlock(192, 128)

#         ############# Block7-up 128-128   ##############
#         self.dense_block7 = BottleneckBlock(128, 64)
#         self.trans_block7 = TransitionBlock(192, 128)

#         ############# Block8-up 256-256 ##############
#         self.dense_block8 = BottleneckBlock(128, 128)
#         self.trans_block8 = TransitionBlock(256, 64)

#         self.conv_refin = nn.Conv2d(64 + 128, 20, 3, 1, 1)
#         self.tanh = nn.Tanh()

#         # T
#         self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

#         # A
#         self.conv2010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv2020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv2030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.conv2040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

#         self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)

#         self.upsample = F.interpolate  # 使用 interpolate 代替 upsample_nearest

#         self.relu = nn.LeakyReLU(0.2, inplace=True)

#         # 全连接层用于估计大气光值
#         self.fc1 = nn.Linear(128, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, 3 * 256 * 256)

#     def forward(self, x):
#         ## 64x64
#         x1 = self.dense_block1(x)
#         x1 = self.trans_block1(x1)

#         ### 16x16
#         x2 = self.trans_block2(self.dense_block2(x1))
#         # print("x2.shape:",x2.shape)
#         ### 16x16
#         x3 = self.trans_block3(self.dense_block3(x2))

#         ### 8x8
#         x4 = self.trans_block4(self.dense_block4(x3))

#         x4 = F.interpolate(x4, size=x2.size()[2:])  # 调整x4的尺寸为x2的尺寸
#         # print("x4.shape:",x4.shape)
#         x42 = torch.cat([x4, x2], 1) # 在通道维度进行拼接
#         # print("x42.shape:", x42.shape)
#         x5 = self.trans_block5(self.dense_block5(x42))

#         x5 = F.interpolate(x5, size=x1.size()[2:])  # 调整x5的尺寸为x1的尺寸
#         x52 = torch.cat([x5, x1], 1)
#         x6 = self.trans_block6(self.dense_block6(x52))

#         x7 = self.trans_block7(self.dense_block7(x6))

#         x8 = self.trans_block8(self.dense_block8(x7))
#         x8 = pad_tensor(x8, 256, 256)  # 对x8进行填充以匹配输出尺寸

#         x8 = torch.cat([x8, F.interpolate(x, size=(256, 256))], 1)  # 将输入调整为256x256并拼接
#         x9 = self.relu(self.conv_refin(x8))

#         shape_out = x9.data.size()[2:4]

#         x101 = F.avg_pool2d(x9, 32)
#         x102 = F.avg_pool2d(x9, 16)
#         x103 = F.avg_pool2d(x9, 8)
#         x104 = F.avg_pool2d(x9, 4)

#         x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
#         x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
#         x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
#         x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

#         dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
#         dehaze = self.tanh(self.refine3(dehaze))

#         x2010 = self.upsample(self.relu(self.conv2010(x101)), size=shape_out)
#         x2020 = self.upsample(self.relu(self.conv2020(x102)), size=shape_out)
#         x2030 = self.upsample(self.relu(self.conv2030(x103)), size=shape_out)
#         x2040 = self.upsample(self.relu(self.conv2040(x104)), size=shape_out)

#         a = torch.cat((x2010, x2020, x2030, x2040, x9), 1)
#         a = self.tanh(self.refine3(a))

#         return dehaze




# class ContrastEnhancement(nn.Module):
#     def __init__(self, kernel_size=3, sigma=1.0):
#         super(ContrastEnhancement, self).__init__()
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
#         self.init_gaussian_kernel()

#     def init_gaussian_kernel(self):
#         weight = self.gaussian_filter.weight
#         weight.data.fill_(0)
#         # Create 2D Gaussian kernel
#         ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
#         xx, yy = torch.meshgrid(ax, ax)
#         kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
#         kernel = kernel / torch.sum(kernel)
#         weight.data[0, 0, :, :] = kernel
    
#     def forward(self, x):
#         # Calculate local contrast
#         x_gray = torch.mean(x, dim=1, keepdim=True)
#         local_mean = self.gaussian_filter(x_gray)
#         local_variance = self.gaussian_filter(x_gray**2) - local_mean**2
#         local_contrast = torch.sqrt(local_variance.clamp(min=1e-5))
        
#         # Enhance contrast
#         enhanced_contrast = x_gray / (local_contrast + 1e-5)
#         return enhanced_contrast.expand_as(x)