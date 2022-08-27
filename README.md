# Deep-learning inversion: A next-generation seismic velocity model building method

This is the python implementation (PyTorch) of the deep leraning model for velocity model building in a surpervised approach. The [paper](https://library.seg.org/doi/10.1190/geo2018-0249.1) is  published on Geophysics. The arxiv version of the paper is availabel 
[here](https://arxiv.org/abs/1902.06267). 

Note that the arxiv version is a litlle different from the publishion, please refer to the official version.

## Abstract

We investigate a novel method based on the supervised deep fully convolutional neural network (FCN) for velocity-model building (VMB) directly from raw seismograms. Unlike the conventional inversion method based on physical models, the supervised deep-learning methods 
are based on big-data training rather than prior-knowledge assumptions. One key characteristic of the deep-learning method is that it 
can automatically extract multi-layer useful features without the need for human-curated activities and initial velocity setup. The 
data-driven method usually requires more time during the training stage, and actual predictions take less time, with only seconds 
needed. Therefore, the computational time of geophysical inversions, including real-time inversions, can be dramatically reduced once a good generalized network is built. 

## Experimental Results
With the goal of estimating velocity models using seismic data as inputs directly, the network needs to project seismic data from the data domain to the model domain. Our method contains two stages: the training process and the prediction process, as shown in the following figure:

![Flowchart of the FCN-based inversion process](/images/schematic.png)

We design the CNN based on the famous [U-Net architecture](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28). We extensively validate the proposed method on similate data and experimental data (i.e. [SEG salt model](https://wiki.seg.org/wiki/Open_data#SEG.2FEAGE_Salt_and_Overthrust_Models)). The following two figures show several visual resultes of our method compared with full waveform inversion (FWI). It denotes that our proposed method is generally feasible for velocity model building.

![Comparisons of the velocity inversion (simulated models)](/images/simulateresult.png)

![Comparisons of the velocity inversion (SEG salt models)](/images/SEGresult.png)

## Dataset

For the training process, we generate the simulated velocity models and their corresponding measurement by solving the acoustic wave equation. We provide the synthetic and SEG salt velocity models for training and testing process which are included in [FCNVMB-data.zip](https://github.com/YangFangShu/FCNVMB-Deep-learning-based-seismic-velocity-model-building). Note that you should generate the corresponding seismic measurement by yourself. Once you have the data pairs, you can train the network as following. 


## Training & Testing

The scripts FCNVMB_train.py and FCNVMB_test.py are implemented for training and testing. If you want to train your own network, please firstly checkout the files named ParamConfig.py and PathConfig.py, to be sure that all the parameters and the paths are consistent, e.g.
```
####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
DataDim       = [2000,301]    # Dimension of original one-shot seismic data
data_dsp_blk  = (5,1)         # Downsampling ratio of input
ModelDim      = [201,301]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 10            # Space interval 


####################################################
####             NETWORK PARAMETERS             ####
####################################################   
BatchSize         = 10        # Number of batch size
LearnRate         = 1e-3      # Learning rate
Nclasses          = 1         # Number of output channels
Inchannels        = 29        # Number of input channels, i.e. the number of shots
SaveEpoch         = 20        
DisplayStep       = 2         # Number of steps till outputting stats
```
and
```
###################################################
####                   PATHS                  #####
###################################################
 
main_dir   = '/home/yfs/Code/pytorch/FCNVMB/'     # Replace your main path here

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

```

Then checkout these two main files to train/test the network, simply type
```
python FCNVMB_train.py
python FCNVMB_test.py
```


## Enviroment Requirement

```
python = 3.8.5
pytorch = 1.4.0
numpy
scipy
matplotlib
scikit-image
math
```
All of them can be installed via ```conda (anaconda)```, e.g.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.0 -c pytorch
```

## Citation

If you find the paper and the code useful in your research, please cite the paper:
```
@article{yang2019deep,
  title={Deep-learning inversion: A next-generation seismic velocity model building method},
  author={Yang, Fangshu and Ma, Jianwei},
  journal={Geophysics},
  volume={84},
  number={4},
  pages={R583--R599},
  year={2019},
  publisher={Society of Exploration Geophysicists}
}
```
If you have any questions about this paper, feel free to contract us: yangfs@hit.edu.cn

