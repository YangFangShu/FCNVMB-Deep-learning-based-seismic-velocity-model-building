# -*- coding: utf-8 -*-
"""
Fully Convolutional neural network (U-Net) for velocity model building from prestack

unmigrated seismic data directly



Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""

################################################
########        IMPORT LIBARIES         ########
################################################

from ParamConfig import *
from PathConfig import *
from LibConfig import *

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device         = torch.device("cuda" if cuda_available else "cpu")

net = UnetModel(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=True) 
if torch.cuda.is_available():
    net.cuda()

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(),lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    premodel_file = models_dir + premodelname + '.pkl'
    ##Load generator parameters
    net  = net.load_state_dict(torch.load(premodel_file))
    net  = net.to(device)
    print('Finish downloading:',str(premodel_file))
    
################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
train_set,label_set,data_dsp_dim,label_dsp_dim  = DataLoad_Train(train_size=TrainSize,train_data_dir=train_data_dir, \
                                                                 data_dim=DataDim,in_channels=Inchannels, \
                                                                 model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                                 label_dsp_blk=label_dsp_blk,start=1, \
                                                                 datafilename=datafilename,dataname=dataname, \
                                                                 truthfilename=truthfilename,truthname=truthname)
# Change data type (numpy --> tensor)
train        = data_utils.TensorDataset(torch.from_numpy(train_set),torch.from_numpy(label_set))
train_loader = data_utils.DataLoader(train,batch_size=BatchSize,shuffle=True)



################################################
########            TRAINING            ########
################################################

print() 
print('*******************************************') 
print('*******************************************') 
print('           START TRAINING                  ') 
print('*******************************************') 
print('*******************************************') 
print() 


print ('Original data dimention:%s'      %  str(DataDim))
print ('Downsampled data dimention:%s '  %  str(data_dsp_dim))
print ('Original label dimention:%s'     %  str(ModelDim))
print ('Downsampled label dimention:%s'  %  str(label_dsp_dim))
print ('Training size:%d'                %  int(TrainSize))
print ('Traning batch size:%d'           %  int(BatchSize))
print ('Number of epochs:%d'             %  int(Epochs))
print ('Learning rate:%.5f'              %  float(LearnRate))
              
# Initialization
loss1  = 0.0
step   = np.int(TrainSize/BatchSize)
start  = time.time()

for epoch in range(Epochs): 
    epoch_loss = 0.0
    since      = time.time()
    for i, (images,labels) in enumerate(train_loader):        
        iteration  = epoch*step+i+1
        # Set Net with train condition
        net.train()
        
        # Reshape data size
        images = images.view(BatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
        labels = labels.view(BatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradient buffer
        optimizer.zero_grad()     
        
        # Forward prediction
        outputs = net(images,label_dsp_dim)
        
        # Calculate the MSE
        loss    = F.mse_loss(outputs,labels,reduction='sum')/(label_dsp_dim[0]*label_dsp_dim[1]*BatchSize)
        
        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')
            
        epoch_loss += loss.item()    
        # Loss backward propagation    
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        # Print loss
        if iteration % DisplayStep == 0:
            print('Epoch: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'.format(epoch+1, \
                                                                               Epochs,iteration, \
                                                                              step*Epochs,loss.item()))        
        
    # Print loss and consuming time every epoch
    if (epoch+1) % 1 == 0:
        #print ('Epoch [%d/%d], Loss: %.10f' % (epoch+1,Epochs,loss.item()))          
        #loss1 = np.append(loss1,loss.item())
        print('Epoch: {:d} finished ! Loss: {:.5f}'.format(epoch+1,epoch_loss/i))
        loss1 = np.append(loss1,epoch_loss/i)
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    # Save net parameters every 10 epochs
    if (epoch+1) % SaveEpoch == 0:
        torch.save(net.state_dict(),models_dir+modelname+'_epoch'+str(epoch+1)+'.pkl')
        print ('Trained model saved: %d percent completed'% int((epoch+1)*100/Epochs))
    

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s' .format(time_elapsed //60 , time_elapsed % 60))

# Save the loss
font2  = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 17,
    }
font3 = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
    }
SaveTrainResults(loss=loss1,SavePath=results_dir,font2=font2,font3=font3)
