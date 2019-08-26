# -*- coding: utf-8 -*-
"""
Path setting

Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""

from LibConfig import *
from ParamConfig import *

####################################################
####                   FILENAMES               ####
####################################################

# Data filename
if SimulateData:
    tagD0 = 'georec'
    tagV0 = 'vmodel'
    tagD1 = 'Rec'
    tagV1 = 'vmodel'
else:
    tagD0 = 'srec'
    tagV0 = 'svmodel'
    tagD1 = 'Rec'
    tagV1 = 'svmodel'

datafilename  = tagD0
dataname      = tagD1
truthfilename = tagV0
truthname     = tagV1



###################################################
####                   PATHS                  #####
###################################################
 
main_dir   = '/home/yfs1/Code/pytorch/FCNVMB/'     # Replace your main path here

## Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')
    
    
## Data path
if os.path.exists('./data/'):
    data_dir    = main_dir + 'data/'               # Replace your data path here
else:
    os.makedirs('./data/')
    data_dir    = main_dir + 'data/'
    
# Define training/testing data directory

train_data_dir  = data_dir  + 'train_data/'        # Replace your training data path here
test_data_dir   = data_dir  + 'test_data/'         # Replace your testing data path here
    
# Define directory for simulate data and SEG data respectively
if SimulateData:
    train_data_dir  = train_data_dir + 'SimulateData/'
    test_data_dir   = test_data_dir  + 'SimulateData/'
else:
    train_data_dir  = train_data_dir + 'SEGSaltData/'
    test_data_dir   = test_data_dir  + 'SEGSaltData/'
    

    
    
## Create Results and Models path
if os.path.exists('./results/') and os.path.exists('./models/'):
    results_dir     = main_dir + 'results/' 
    models_dir      = main_dir + 'models/'
else:
    os.makedirs('./results/')
    os.makedirs('./models/')
    results_dir     = main_dir + 'results/'
    models_dir      = main_dir + 'models/'
    
if SimulateData:
    results_dir     = results_dir + 'SimulateResults/'
    models_dir      = models_dir  + 'SimulataModel/'
else:
    results_dir     = results_dir + 'SEGSaltResults/'
    models_dir      = models_dir  + 'SEGSaltModel/'
    
if os.path.exists(results_dir) and os.path.exists(models_dir):  
    results_dir     = results_dir
    models_dir      = models_dir 
else:
    os.makedirs(results_dir)
    os.makedirs(models_dir)
    results_dir     = results_dir
    models_dir      = models_dir

# Create Model name
if SimulateData:
    tagM = 'Simulate'
else:
    tagM = 'SEGSalt'
tagM0 = '_FCNVMBModel'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch'     + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR'        + str(LearnRate)

modelname = tagM+tagM0+tagM1+tagM2+tagM3+tagM4
# Change here to set the model as the pre-trained initialization
premodelname = 'Simulate_FCNVMBModel_TrainSize1600_Epoch100_BatchSize10_LR0.001'   

