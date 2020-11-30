# -*- coding: utf-8 -*-
"""
Load testing data set

Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io


def DataLoad_Test(test_size,test_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname):
    for i in range(start,start+test_size):
        filename_seis = test_data_dir+'georec_test/'+datafilename+str(i)
        print (filename_seis)
        data1_set = scipy.io.loadmat(filename_seis)
        data1_set = np.float32(data1_set[str(dataname)].reshape([data_dim[0],data_dim[1],in_channels]))
        for k in range (0,in_channels):
            data11_set     = np.float32(data1_set[:,:,k])
            data11_set     = np.float32(data11_set)
            data11_set     = block_reduce(data11_set,block_size=data_dsp_blk,func=decimate)
            data_dsp_dim   = data11_set.shape
            data11_set     = data11_set.reshape(1,data_dsp_dim[0]*data_dsp_dim[1])
            if k==0:
                test1_set  = data11_set
            else:
                test1_set  = np.append(test1_set,data11_set,axis=0)
            filename_label = test_data_dir+'vmodel_test/'+truthfilename+str(i)
            data2_set      = scipy.io.loadmat(filename_label)
            data2_set      = np.float32(data2_set[str(truthname)].reshape(model_dim))
            data2_set      = block_reduce(data2_set,block_size=label_dsp_blk,func=np.max)
            label_dsp_dim  = data2_set.shape
            data2_set      = data2_set.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
            data2_set      = np.float32(data2_set)
            if i==start:
                test_set   = test1_set
                label_set  = data2_set
            else:
                test_set   = np.append(test_set,test1_set,axis=0)
                label_set  = np.append(label_set,data2_set,axis=0)

    test_set  = test_set.reshape((test_size,in_channels,data_dsp_dim[0]*data_dsp_dim[1]))
    label_set = label_set.reshape((test_size,label_dsp_dim[0]*label_dsp_dim[1]))

    
    return test_set, label_set, data_dsp_dim, label_dsp_dim

# downsampling function by taking the middle value
def decimate(a,axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1,-1)]+1.0)/2.0-1).reshape(-1)
    downa = np.array(a)[:,:,idx[0].astype(int),idx[1].astype(int)]
    return downa
