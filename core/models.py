import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
#from .blocks import M_Encoder
#from .blocks import M_Decoder
#from .blocks import M_Conv
import cv2

from core.FastGuidedFilter import GuidedFilter

class M_Net(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(M_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, xgrey):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')

        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class UNet512 (nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(UNet512, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, xgrey):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
                                      #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)

        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        # out = torch.squeeze(out, dim=1)
        return [out, out, out, out, out]

class UNet512_sideoutput(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(UNet512_sideoutput, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, xgrey):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
                                      #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)

        out = self.up5(down5, out)
        sout5 = self.side_5(out)

        out = self.up4(down4, out)
        sout6 = self.side_6(out)

        out = self.up3(down3, out)
        sout7 = self.side_7(out)

        out = self.up2(down2, out)
        sout8 = self.side_8(out)

        out = self.classify(out)

        # out = torch.squeeze(out, dim=1)
        return [out, sout5, sout6, sout7, sout8]

class SU_Net(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(SU_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.changeU1 = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, padding=1)
        self.changeU2 = nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, padding=1)
        self.changeU3 = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, padding=1)
        self.changeU4 = nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, padding=1)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.change1 = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, padding=1)
        self.change2 = nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, padding=1)
        self.change3 = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, padding=1)
        self.change4 = nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, padding=1)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        #self.gf = SGGuidedFilter(r=2, eps=1e-2)
        self.gf = GuidedFilter(r=2, eps=1e-2)

    def forward(self, x, ImgGreys):
        _, _, img_shape, _ = x.size()

        ImgGreys_2 = F.upsample(ImgGreys, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        ImgGreys_3 = F.upsample(ImgGreys, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        ImgGreys_4 = F.upsample(ImgGreys, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')

        conv1, out = self.down1(x)
        up1_1 = self.gf(ImgGreys_2, out)
        out = torch.cat([up1_1, out], dim=1)
        #out = self.changeU1(out)

        conv2, out = self.down2(out)
        #out = torch.cat([self.conv3(x_3), out], dim=1)
        up2_1 = self.gf(ImgGreys_3, out)
        out = torch.cat([up2_1, out], dim=1)

        conv3, out = self.down3(out)
        #out = torch.cat([self.conv4(x_4), out], dim=1)
        up3_1 = self.gf(ImgGreys_4, out)
        out = torch.cat([up3_1, out], dim=1)

        conv4, out = self.down4(out)
        out = self.center(out)

        up5 = self.up5(conv4, out)
        up5_1 = self.gf(ImgGreys_4, up5)
        up5_2 = torch.cat([up5, up5_1], dim=1)
        up5 = self.change1(up5_2)

        up6 = self.up6(conv3, up5)
        up6_1 = self.gf(ImgGreys_3, up6)
        up6_2 = torch.cat([up6, up6_1], dim=1)
        up6 = self.change2(up6_2)

        up7 = self.up7(conv2, up6)
        up7_1 = self.gf(ImgGreys_2, up7)
        up7_2 = torch.cat([up7, up7_1], dim=1)
        up7 = self.change3(up7_2)

        up8 = self.up8(conv1, up7)
        up8_1 = self.gf(ImgGreys, up8)
        up8_2 = torch.cat([up8, up8_1], dim=1)
        up8 = self.change4(up8_2)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8]