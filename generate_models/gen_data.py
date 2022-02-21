# %%
import random
import numpy as np
import numpy.random as random2
import matplotlib.pylab  as plt
import matplotlib as mpl
from  scipy.ndimage import gaussian_filter
from scipy import signal
import fwi
import torch
import time 
from model_generator import  create_random_model, create_random_model2
from skimage.transform import resize
import sys 
import os



# ========================== Functions  ============================== #
def freq_filter(freq, wavelet,btype,fs):
    """
    Filter out low frequency

    Parameters
    ----------
    freq : :obj:`int` or `array in case of bandpass `
    Cut-off frequency
    wavelet : :obj:`torch.Tensor`
    Tensor of wavelet
    btype : obj: 'str'
    Filter type  
    dt : :obj:`float32`
    Time sampling
    Returns
    -------
    : :obj:`torch.Tensor`
    Tensor of highpass frequency wavelet
    """

    if btype == 'hp': sos = signal.butter(4,  2 * freq /fs, 'hp', output='sos') 
    if btype == 'lp': sos = signal.butter(4,   2 * freq /fs , 'lp', output='sos') 
    if btype == 'bp': sos = signal.butter(4,  [2 * freq[0] /fs,  2 * freq[1] /fs ], 
                            'bp', output='sos') 
    return torch.tensor( signal.sosfiltfilt(sos, wavelet,axis=0).copy(),dtype=torch.float32)





# ============================ setting global parameters =============================#
freq=8  # Ricker dominant freq for the source
minimum_freq= 7 # minimum frequency for the data 
nx = 600
nz = 200
dx = 0.02
dt = 0.002
fs = 1/dt # sampling frequency 
nt = int( 5 / dt)  
num_dim=2 
num_shots = 2
num_sources_per_shot=1

num_batches = 1 
mxoffset = 5
num_receiver_per_shot= int((mxoffset/dx)//2)
alphatv=0.001  #  TV regularization coefficient 
source_spacing = (nx*dx - 2*mxoffset -1)/num_shots
receivers_depth = dx
source_depth = dx
osou=0 
orec=0
receiver_spacing= dx*2
FWI_itr= 200
istart=int(sys.argv[1]) * int(sys.argv[2]) 
num_models=int (sys.argv[2])

device = torch.device('cuda:0')
path = './output/'

if not os.path.exists(path):
    os.makedirs(path) 




# ===================================== create random models ========================================= #

# reading the background velocity 
bp = np.load('bp_left.npy')
bp = bp.T
bp = resize(bp,(bp.shape[0],nz))
bp[ bp >= 4.4] = np.nan
# Compute mean and std
bp_mean = np.nanmean(bp,axis=0)
bp_std = np.mean(np.nanstd(bp,axis=0))

models1D = np.zeros((num_models,nz))
initials1D = np.zeros((num_models,nz))
wb = np.zeros((num_models,1))
for i in range(num_models):
	layer = int(np.random.rand()*nz)  
	models1D[i,:], initials1D[i,:],wb[i,:] = create_random_model2(layer,nz,bp_mean,std=bp_std)



# convert to 2D models 
models2D = np.repeat(models1D,nx,axis=1)
models2D = np.reshape(models2D,(num_models,nz,nx))

initials2D = np.repeat(initials1D,nx,axis=1)
initials2D = np.reshape(initials2D,(num_models,nz,nx))





# convert model to tensors
true_m = torch.tensor(models2D,dtype=torch.float32)
init_m = torch.tensor(initials2D,dtype=torch.float32) 

# %%
# ====================================== Inversion ======================================= #
start = time.time()
inversion = fwi.fwi(nx,nz,dx,nt,dt,
		num_dim,num_shots,num_sources_per_shot,source_spacing,osou,num_receiver_per_shot,receiver_spacing,
        orec,source_depth,receivers_depth,num_batches,2)
wavel = inversion.Ricker(freq)  #source will be repeated as many shots

# high-pass filter for removing low frequencies
wavel_f = freq_filter(freq=minimum_freq,wavelet=wavel,btype='hp',fs=fs)

# forward modelling
data = torch.zeros((nt,num_shots,num_receiver_per_shot),dtype=torch.float32)
inverted_models = torch.zeros((nz,nx),dtype=torch.float32)
count=1

for i in  range(num_models): 
     print("MODEL NUMBER: ", i)

     data  = inversion.forward_modelling(true_m[i,:],wavel_f.repeat(1,num_shots,num_sources_per_shot),device) 

     inverted_models  = inversion.run_inversion (init_m[i,:],
                        data,wavel_f,wb[i,:],FWI_itr,alphatv,device) # output is numpy arr
     np.save(path+'true_m'+str(istart+count),true_m[i,:,0].cpu().numpy())
     np.save(path+'init_m'+str(istart+count),init_m[i,:,0].cpu().numpy())
     np.save(path+'inv_m'+str(istart+count),inverted_models[:,:,0])
     count += 1 


end = time.time()
print ('Running time  for all the inversino is :', (end-start)/60, ' minutes')
print (f"Number of generated file is {count-1}")

print(' ---------- Done yaaaay')

