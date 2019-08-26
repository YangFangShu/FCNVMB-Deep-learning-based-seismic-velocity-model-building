# -*- coding: utf-8 -*-
"""
Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""
import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable


def turn(GT):
    dim = GT.shape
    for j in range(0,dim[1]):
        for i in range(0,dim[0]//2):
            temp    = GT[i,j]
            GT[i,j] = GT[dim[0]-1-i,j]
            GT[dim[0]-1-i,j] = temp
    return GT 


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target     = Variable(torch.from_numpy(target))
    zero       = torch.zeros_like(target)   
    criterion  = nn.MSELoss(size_average=True)    
    MSE        = criterion (prediction, target)
    total      = criterion (target, zero)
    psnr       = 10. * log10(total.item() / MSE.item())
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def SaveTrainResults(loss,SavePath,font2,font3):
    fig,ax  = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('MSE Loss', font2)
    ax.set_title('Training', font3)
    ax.set_xlim([1,6])
    ax.set_xticklabels(('0','20','40','60','80','100'))
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed',linewidth=0.5)
    
     
    plt.savefig(SavePath+'TrainLoss', transparent = True)
    data = {}
    data['loss'] = loss
    scipy.io.savemat(SavePath+'TrainLoss',data)
    plt.show(fig)
    plt.close()

def SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM    
    data['GT']      = GT
    data['Prediction'] = Prediction
    scipy.io.savemat(SavePath+'TestResults',data) 
    
    
def PlotComparison(pd,gt,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    PD = pd.reshape(label_dsp_dim[0],label_dsp_dim[1])
    GT = gt.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig1,ax1 = plt.subplots(figsize=(6, 4))    
    im1     = ax1.imshow(GT,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1    = divider.append_axes("right",size="5%",pad=0.05)
    plt.colorbar(im1,ax=ax1,cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)',font2)
    ax1.set_ylabel('Depth (km)',font2)
    ax1.set_title('Ground truth',font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'GT',transparent=True)
    
    fig2,ax2=plt.subplots(figsize=(6, 4))
    im2=ax2.imshow(PD,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)

    plt.tick_params(labelsize=12)  
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(14)   
    ax2.set_xlabel('Position (km)',font2)
    ax2.set_ylabel('Depth (km)',font2)
    ax2.set_title('Prediction',font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15,top=0.92,left=0.08,right=0.98)
    plt.savefig(SavePath+'PD',transparent=True)
    plt.show(fig1)
    plt.show(fig2)
    plt.close()
   