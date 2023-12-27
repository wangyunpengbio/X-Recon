
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F



##########################################
'''
3D Layers
'''
##########################################
'''
Define a Up-sample block
Network:
  x c*d*h*w->
  -> Up-sample s=2
  -> 3*3*3*c1 or 1*1*1*c1 stride=1 padding conv
  -> norm_layer, activation
'''

# 这个作为decoder用到了
class Upsample_3DUnit(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel, norm_layer, scale_factor=2, upsample_mode='nearest',
                 activation=nn.ReLU(True), use_bias=True):
        super(Upsample_3DUnit, self).__init__()
        if upsample_mode == 'trilinear' or upsample_mode == 'nearest':
            self.block = Upsample_3DBlock(kernel_size, input_channel, output_channel, norm_layer, scale_factor,
                                          upsample_mode, activation, use_bias)
        elif upsample_mode == 'transposed':
            self.block = Upsample_TransposedConvBlock(kernel_size, input_channel, output_channel, norm_layer,
                                                      scale_factor, activation, use_bias)
        else:
            raise NotImplementedError()

    def forward(self, input):
        return self.block(input)

# 被Upsample_3DUnit调用
class Upsample_3DBlock(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel, norm_layer,
                 scale_factor=2, upsample_mode='nearest', activation=nn.ReLU(True), use_bias=True):
        super(Upsample_3DBlock, self).__init__()
        conv_block = []
        conv_block += [nn.Upsample(scale_factor=scale_factor, mode=upsample_mode),
                       nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, padding=int(kernel_size // 2),
                                 bias=use_bias),
                       norm_layer(output_channel),
                       activation]
        self.block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.block(input)

# 被Upsample_3DUnit调用
class Upsample_TransposedConvBlock(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel, norm_layer, scale_factor=2, activation=nn.ReLU(True),
                 use_bias=True):
        super(Upsample_TransposedConvBlock, self).__init__()
        conv_block = []
        conv_block += [
            nn.ConvTranspose3d(input_channel, output_channel, kernel_size=kernel_size, padding=int(kernel_size // 2),
                               bias=use_bias, stride=scale_factor, output_padding=int(kernel_size // 2)),
            norm_layer(output_channel),
            activation
        ]
        self.block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.block(input)

'''
Define a 2D Dense block
Network:
  x ->
  -> for i in num_layers:
        -> norm relu 1*1*tg conv 
        -> norm relu 3*3*g conv
      norm
'''


# 这个作为基础模块用到了，被Dense_2DBlock调用
class _DenseLayer2D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer, activation, use_bias):
        super(_DenseLayer2D, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', activation),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=use_bias)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', activation),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=use_bias)),

    def forward(self, x):
        new_features = super(_DenseLayer2D, self).forward(x)
        return torch.cat([x, new_features], 1)


 # 这个作为基础模块用到了
class Dense_2DBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=16, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(True), use_bias=True):
        super(Dense_2DBlock, self).__init__()
        conv_block = []
        for i in range(num_layers):
            conv_block += [
                _DenseLayer2D(num_input_features + i * growth_rate, growth_rate, bn_size, norm_layer, activation,
                              use_bias)]
        self.conv_block = nn.Sequential(*conv_block)
        self.next_input_features = num_input_features + num_layers * growth_rate
        # self.conv_block.add_module('finalnorm', norm_layer(self.next_input_features))

    def forward(self, input):
        return self.conv_block(input)


##########################################
'''
2D To 3D Layers
'''

# 作为link_layers使用
##########################################
class Dimension_UpsampleCutBlock(nn.Module):
    def __init__(self, input_channel, output_channel, norm_layer2d, norm_layer3d, activation=nn.ReLU(True),
                 use_bias=True):
        super(Dimension_UpsampleCutBlock, self).__init__()

        self.output_channel = output_channel
        compress_block = [
            nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0, bias=use_bias),
            norm_layer2d(output_channel),
            activation
        ]
        self.compress_block = nn.Sequential(*compress_block)

        conv_block = []
        conv_block += [
            nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=use_bias),
            norm_layer3d(output_channel),
            activation,
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        # input's shape is [NCHW]
        N, _, H, W = input.size()
        # expand to [NCDHW]
        return self.conv_block(self.compress_block(input).unsqueeze(2).expand(N, self.output_channel, H, H, W))


##########################################
'''
View Fusion Layers / Functions
'''

# 作为transposed_layer使用
##########################################
class Transposed_And_Add(nn.Module):
    def __init__(self, view1Order, view2Order, sortOrder=None):
        super(Transposed_And_Add, self).__init__()
        self.view1Order = view1Order
        self.view2Order = view2Order
        self.permuteView1 = tuple(np.argsort(view1Order))
        self.permuteView2 = tuple(np.argsort(view2Order))

    def forward(self, *input):
        # return tensor in order of sortOrder
        return (input[0].permute(*self.permuteView1) + input[1].permute(*self.permuteView2))
