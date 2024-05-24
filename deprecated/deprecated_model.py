
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
