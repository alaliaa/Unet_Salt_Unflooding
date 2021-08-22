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
import m8r as sf
import sys 

# Function for bandpass filtering 
def butter_bandpass(lowcut, highcut, fs,btype ,order=7):
    nyq = 0.5 * fs
    if lowcut != None: low = lowcut / nyq
    if highcut != None: high = highcut / nyq

    if btype == 'band': b, a = signal.butter(order, [low, high], btype=btype, analog=False)
    if btype == 'low':  b, a = signal.butter(order, low, btype=btype, analog=False)
    if btype == 'high': b, a = signal.butter(order,high, btype=btype, analog=False)
    
    return b, a        


def butter_bandpass_filter(data,lowcut=None,highcut=None,fs=None,btype='band',order=5):
    b, a = butter_bandpass(lowcut, highcut, fs,btype, order=order)
    y = signal.filtfilt(b, a, data)
    return y 


def load_2drsf_data(filename):
    f  = sf.Input(filename)
    nz = f.int("n1")
    nx = f.int("n2")
    dz = f.float("d1")
    dx = f.float("d2")
    oz = f.float("o1")
    ox = f.float("o2")

    # note in reading rsf to numpy the diload_rsf_datamension are reverse 
    data = np.zeros((nx,nz),dtype=np.float32)
    f.read(data)
    print('Shape of loaded data: {}'.format(np.shape(data)))
    parm = {'nz':nz, 'nx':nx, 
            'dz':dz, 'dx':dx, 
            'oz':oz, 'ox':ox
            }
    return data.T,parm

# %% 
# setting parameters

mpl.rcParams['image.cmap'] ='seismic'

freq=8
nx = 600
nz = 200
dx = 0.02
dt = 0.002
fs = 1/dt # sampling frequency 
nt = int( 5 / dt)  
num_dim=2 
num_shots = 2
num_sources_per_shot=1
# source_spacing= (nx*dx)/num_shots
# source_spacing=6.8
num_batches = 1 
mxoffset = 5
num_receiver_per_shot= int((mxoffset/dx)//2)
alphatv=0.001

source_spacing = (nx*dx - 2*mxoffset -1)/num_shots
receivers_depth = .02
source_depth = .02
os=0
orec=0
receiver_spacing= dx*2
FWI_itr= 200
istart=int(sys.argv[1]) * int(sys.argv[2]) 
num_models=int (sys.argv[2])


device = torch.device('cuda:0')
path = './output_ibx5/'
# path = './output/test/'




# get 1D models 
bp, par = load_2drsf_data('./model_scale_1.rsf')
bp = bp.T
bp = resize(bp,(bp.shape[0],nz))

bp[ bp >= 4.4] = np.nan
# Compute mean and std
bp_mean = np.nanmean(bp,axis=0)

models1D = np.zeros((num_models,nz))
initials1D = np.zeros((num_models,nz))
wb = np.zeros((num_models,1))
for i in range(num_models):
	layer = int(np.random.rand()*nz)  # 80 is max numpur of layer
	models1D[i,:], initials1D[i,:],wb[i,:] = create_random_model2(layer,nz,bp_mean)



# convert to 2D models 
models2D = np.repeat(models1D,nx,axis=1)
models2D = np.reshape(models2D,(num_models,nz,nx))

initials2D = np.repeat(initials1D,nx,axis=1)
initials2D = np.reshape(initials2D,(num_models,nz,nx))





# convert model to tensors
true_m = torch.tensor(models2D,dtype=torch.float32)
init_m = torch.tensor(initials2D,dtype=torch.float32) 

# %%
# inversion blocj 
start = time.time()
inversion = fwi.fwi(nx,nz,dx,nt,dt,
		num_dim,num_shots,num_sources_per_shot,source_spacing,os,num_receiver_per_shot,receiver_spacing,
        orec,source_depth,receivers_depth,num_batches,2)
wavel = inversion.Ricker(freq)  #source will be repeated as many shots

# filter frequency
wavel_f = torch.from_numpy(butter_bandpass_filter(
			wavel.view(wavel.shape[0]).numpy(),lowcut=None,highcut=7,fs=fs,btype='high').copy()).view(wavel.shape[0],1,1)
    

# forward modelling

data = torch.zeros((nt,num_shots,num_receiver_per_shot),dtype=torch.float32)
inverted_models = torch.zeros((nz,nx),dtype=torch.float32)
count=1

for i in  range(num_models): 
     print("MODEL NUMBER: ", i)

     data  = inversion.forward_modelling(true_m[i,:],wavel_f.repeat(1,num_shots,num_sources_per_shot),device) 

    #  plt.figure()
    #  vmin, vmax = np.percentile(data[:,0,:].numpy(), [2,98])
    #  plt.imshow(data[:,0,:].cpu().numpy(), aspect='auto',vmin=vmin, vmax=vmax)
    #  plt.show()



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

# plot one shot gather


# inverted_models = inverted_models.numpy()

#for i in range (num_models):
#     plt.figure()
#     plt.plot(true_m[i,:,5].numpy(),label='true')     
#     plt.plot(init_m[i,:,5].numpy(),label='initial')
#     plt.plot(inverted_models[i,:,5],label='inv')a
#     plt.legend()
#     plt.show()
# %.
