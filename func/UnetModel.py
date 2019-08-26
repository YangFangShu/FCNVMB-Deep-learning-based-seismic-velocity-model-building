# -*- coding: utf-8 -*-
"""
Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""

################################################
########        DESIGN   NETWORK        ########
################################################

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # Skip and concatenate 
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class  UnetModel(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm):
        super(UnetModel, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        
        filters = [64, 128, 256, 512, 1024]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4     = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3     = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2     = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1     = unetUp(filters[1], filters[0], self.is_deconv)
        self.final   = nn.Conv2d(filters[0],self.n_classes, 1)
        
    def forward(self, inputs,label_dsp_dim):
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        center = self.center(down4)
        up4    = self.up4(down4, center)
        up3    = self.up3(down3, up4)
        up2    = self.up2(down2, up3)
        up1    = self.up1(down1, up2)
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return self.final(up1)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

