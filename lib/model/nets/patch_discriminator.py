
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import functools

from .utils import CoordConv3D

'''
Patch Discriminator
'''

class NLayer_2D_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, n_out_channels=1):
        super(NLayer_2D_Discriminator, self).__init__()

        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        sequence = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                          nn.Sigmoid()]]
        else:
            sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return [self.model(input)]

 # 在第一层加入Coord
class NLayer_3D_Discriminator_CoordConv(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False,
                 n_out_channels=1):  # 本次实验中discriminator_feature为False，即getIntermFeat为False
        super(NLayer_3D_Discriminator_CoordConv, self).__init__()

        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        sequence = [[
            CoordConv3D(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                          nn.Sigmoid()]]
        else:
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return [self.model(input)]


class NLayer_3D_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,  # 本次实验中discriminator_feature为False，即getIntermFeat为False
                 norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False, n_out_channels=1):
        super(NLayer_3D_Discriminator, self).__init__()

        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        sequence = [[
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid:
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                          nn.Sigmoid()]]
        else:
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return [self.model(input)]

# 来源：https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/pytorch/graphs/models/discriminator.py
class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet3D, self).__init__()
        self.input_channel = in_channel
        self.num_classes = n_classes
        kernel_size = (3,3,3)
        kernel_size_deconv = (2,2,2)
        stride_deconv = (2,2,2)
        out_channels = 4 # 可以作为ndf # 原来是32
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout3d(p=0.2, inplace=False)

        # Defining the convolutional operations
        self.pool = nn.AvgPool3d(2)

        self.encoder0 = nn.Conv3d(self.input_channel, out_channels, kernel_size, padding="same")
        self.encoder1 = nn.Conv3d(out_channels, out_channels, kernel_size, padding="same")

        self.encoder2 = nn.Conv3d(out_channels, out_channels*(2), kernel_size, padding="same")
        self.encoder3 = nn.Conv3d(out_channels*(2), out_channels*(2), kernel_size, padding="same")

        self.encoder4 = nn.Conv3d(out_channels*(2), out_channels*(2**2), kernel_size, padding="same")
        self.encoder5 = nn.Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size, padding="same")

        self.encoder6 = nn.Conv3d(out_channels*(2**2), out_channels*(2**3), kernel_size, padding="same")
        self.encoder7 = nn.Conv3d(out_channels*(2**3), out_channels*(2**3), kernel_size, padding="same")

        self.decoder1 = nn.ConvTranspose3d(out_channels*(2**3), out_channels*(2**3), kernel_size_deconv, stride_deconv)
        #encoder5 + decoder1
        self.encoder8 = nn.Conv3d(out_channels*(2**2) + out_channels*(2**3), out_channels*(2**2), kernel_size, padding="same")
        self.encoder9 = nn.Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size, padding="same")

        self.decoder2 = nn.ConvTranspose3d(out_channels*(2**2), out_channels*(2**2), kernel_size_deconv, stride_deconv)
        #encoder3 + decoder2
        self.encoder10 = nn.Conv3d(out_channels*(2) + out_channels*(2**2), out_channels*(2), kernel_size, padding="same")
        self.encoder11 = nn.Conv3d(out_channels*(2), out_channels*(2), kernel_size, padding="same")

        self.decoder3 = nn.ConvTranspose3d(out_channels*(2), out_channels*(2), kernel_size_deconv, stride_deconv)
        #encoder1 + decoder3
        self.encoder12 = nn.Conv3d(out_channels + out_channels*(2), out_channels, kernel_size, padding="same")
        self.encoder13 = nn.Conv3d(out_channels, out_channels, kernel_size, padding="same")

        self.final_conv = nn.Conv3d(out_channels, self.num_classes, kernel_size)

    def forward(self, x, use_dropout=False):
        x = self.lrelu(self.encoder0(x))
        conv1 = self.lrelu(self.encoder1(x))
        x = self.pool(conv1)

        x = self.lrelu(self.encoder2(x))
        conv3 = self.lrelu(self.encoder3(x))
        x = self.pool(conv3)

        x = self.lrelu(self.encoder4(x))
        conv5 = self.lrelu(self.encoder5(x))
        x = self.pool(conv5)

        x = self.lrelu(self.encoder6(x))
        x = self.lrelu(self.encoder7(x))

        if use_dropout:
            x = self.dropout(x)

        deconv1 = self.decoder1(x)

        x = torch.cat((conv5, deconv1), 1)
        del conv5, deconv1
        x = self.lrelu(self.encoder8(x))
        x = self.lrelu(self.encoder9(x))

        deconv2 = self.decoder2(x)
        x = torch.cat((conv3, deconv2), 1)
        del conv3, deconv2
        x = self.lrelu(self.encoder10(x))
        x = self.lrelu(self.encoder11(x))

        deconv3 = self.decoder3(x)
        x = torch.cat((conv1, deconv3), 1)
        del conv1, deconv3
        x = self.lrelu(self.encoder12(x))
        x = self.lrelu(self.encoder13(x))

        if use_dropout:
            x = self.dropout(x)

        final_output = self.final_conv(x)

        return final_output


# 还没改成Residual模式，目前只是把D对齐了，效果居然会变差
class NLayer_3D_Discriminator_CoordConv_PaddingMatch(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False,
                 n_out_channels=1):  # 本次实验中discriminator_feature为False，即getIntermFeat为False
        super(NLayer_3D_Discriminator_CoordConv_PaddingMatch, self).__init__()

        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        padw = "same" # 这样就是一一对应的关系
        padw_2 = int((kw - 2.0)/2)

        sequence = [[
            CoordConv3D(input_nc, ndf, kernel_size=kw, stride=2, padding=padw_2), # 原来会变成128/2+1=65,feature map 为65*65
            nn.LeakyReLU(0.2, True)
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw_2, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        if use_sigmoid: # 不进行sigmoid
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                          nn.Sigmoid()]]
        else:
            sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return [self.model(input)]

'''
Multi-Scale
Patch Discriminator
'''


#############################################################
# 3D Version
#############################################################
# num_D数量的NLayer_3D_Discriminator，分别接收多个尺度的input，然后把结果append到一起去。
class Multiscale_3D_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False, num_D=3, n_out_channels=1):
        super(Multiscale_3D_Discriminator, self).__init__()
        assert num_D >= 1
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayer_3D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat,
                                           n_out_channels)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))

            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result


#############################################################
# 2D Version
#############################################################
class Multiscale_2D_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False, num_D=3, n_out_channels=1):
        super(Multiscale_2D_Discriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayer_2D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat,
                                           n_out_channels)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))

            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result
