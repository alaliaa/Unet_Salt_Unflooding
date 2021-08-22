import copy
import random
import sys
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter1d
from skimage.transform import resize
from torch.utils.data import DataLoader
import  m8r as sf
import os 
import subprocess

def Plot_model(m,par,name=None):
    font = {
        'weight' : 'bold',
        'size'   : 18}
    mlp.rc('font', **font)    
    
    vmin, vmax = np.percentile(m.T,[2,98])
    plt.figure(figsize=(10,3))
    plt.imshow(m,cmap='jet',vmin=vmin,vmax=vmax,extent=[par['ox'],par['dx']*par['nx'],par['nz']*par['dz'],par['oz']])
    plt.axis('tight')
    plt.xlabel('Distance (km)',fontsize=20,weight='heavy')
    plt.ylabel('Depth (km)',fontsize=20,weight='heavy')
    if name!=None: plt.savefig('./output/figure/'+name,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

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

def plot_models1D(inp,label,num_models,i,j):
    n=i*j
    indx = []
    f = plt.figure(figsize=(60,30))
    font = {
        'weight' : 'bold',
        'size'   : 30}
    mlp.rc('font', **font)
    for k in range(1,n+1):    
        ax = f.add_subplot(i,j,k)
        m  = random.randint(0,num_models-1)
        im =ax.plot(inp[m,:],label='Input FWI',linewidth=6)
        im =ax.plot(label[m,:],label='Target',linewidth=6)
        plt.axis('tight')
        plt.legend(prop={'size': 20, 'weight':'bold'},loc='upper left')
    plt.savefig('./output/figure/inplabels',bbox_inches='tight')
    plt.close()
    
    
def plot_prediction(X,Y,pred,init,i,j):
    num_models = X.shape[0]
    n=i*j

    # f = plt.figure(figsize=(30,6))
    f = plt.figure(figsize=(60,50))
    font = {
        'weight' : 'heavy',
        'size'   : 23}
    mlp.rc('font', **font)

    m  = np.random.randint(low=0,high=num_models-1,size=n)
    # m  = [709, 1296, 101]

    print(m)
    indx=0

    for k in range(1,n+1):    

        ax = f.add_subplot(i,j,k)
        #m  = random.randint(0,num_models-1)
        #print('model number',m[indx])
        ax.plot(np.arange(0,200)*0.02,X[m[indx],:],label='Input',linewidth=4,color='black' )
        ax.plot(np.arange(0,200)*0.02,init[m[indx],:],'--',label='Initial for FWI',linewidth=4,color='green' )
        ax.plot(np.arange(0,200)*0.02,Y[m[indx],:],label='Target',linewidth=6,color='blue')
        ax.plot(np.arange(0,200)*0.02,pred[m[indx],:],label='Prediction',linewidth=4,color='red')

#           plt.axis('tight')
        plt.legend(prop={'size': 22, 'weight':'heavy'},loc='upper left')
        plt.xlabel('Depth (km)',weight='heavy')
        plt.ylabel('Velocity (km/s)',weight='heavy')
        indx +=1
        #plt.legend()
        #plt.title('Validation')
    plt.show()
  
def plot_history(trainingloss,testingloss,netname):

    #trainingloss = np.array(trainingloss)
    #testingloss =  np.array(testingloss)
    Epc = np.arange(1,trainingloss.shape[0]+1)
    plt.figure(figsize=(8,6))
    plt.semilogy(trainingloss,color='b',label='Training')
    plt.semilogy(testingloss,color='r',label='Validation')

    plt.xlabel('Epochs', fontsize=24, fontweight='bold')
    plt.ylabel('Loss ',fontsize=24, fontweight='bold')
    plt.legend()
    plt.tick_params(axis='both',which='minor',labelsize=20)
    plt.xticks(fontsize=20,fontweight='semibold')
    plt.yticks(fontsize=20,fontweight='semibold')
    plt.legend(prop={'size': 20, 'weight':'bold'})

    name='./output/figure/Loss_'+netname+'.png'
    plt.savefig(name, bbox_inches='tight')




def plot_r2(trainingloss,testingloss,netname):

    #trainingloss = np.array(trainingloss)
    #testingloss =  np.array(testingloss)
    Epc = np.arange(1,trainingloss.shape[0]+1)
    plt.figure(figsize=(8,6))
    plt.plot(trainingloss,color='b',label='Training')
    plt.plot(testingloss,color='r',label='Validation')

    plt.xlabel('Epochs', fontsize=24, fontweight='bold')
    plt.ylabel('R2',fontsize=24, fontweight='bold')
    plt.legend()
    plt.tick_params(axis='both',which='minor',labelsize=20)
    plt.xticks(fontsize=20,fontweight='semibold')
    plt.yticks(fontsize=20,fontweight='semibold')
    plt.legend(prop={'size': 20, 'weight':'bold'})

    name='./output/figure/R2_'+netname+'.png'
    plt.savefig(name, bbox_inches='tight')
    
# def scale_data(data=None):

#     scaler= StandardScaler().fit(data)
#     #scaler =  MinMaxScaler()
#     # scaler = MaxAbsScaler()
#     #scaler = RobustScaler()
#     # scaler.fit(data)
#     print(f'scaler mean {scaler.mean_} with the shape {scaler.mean_.shape}')
#     print(f'scaler scale {scaler.scale_}')
#     scaled_data = scaler.transform(data)
#     return scaled_data.T,scaler


def get_scaler(data=None):
    ''' this function used to get scaler from training set'''
    #scaler= StandardScaler().fit(data)
    #scaler =  MinMaxScaler((-1,1)).fit(data)
    scaler = MaxAbsScaler().fit(data)
    # scaler = RobustScaler().fit(data)

    return scaler


def scale_data(data=None,scaler=None,mode='forward'):
    if mode == 'forward':scaled_data = scaler.transform(data)
    elif mode == 'inv': scaled_data = scaler.inverse_transform(data)
    else: sys.exit("Error!: scaling mode is not 'forward' nor 'inv' " )
    
    return scaled_data



def plot_models1D2(inp,label,init,num_models,i,j):
    n=i*j
    indx = []
    f = plt.figure(figsize=(60,30))
    font = {
        'weight' : 'bold',
        'size'   : 30}
    mlp.rc('font', **font)
    for k in range(1,n+1):    
        ax = f.add_subplot(i,j,k)
        m  = random.randint(0,num_models-1)
        im =ax.plot(inp[m,:],label='Input FWI',linewidth=6,color='blue')
        im =ax.plot(label[m,:],label='True model (target)',linewidth=6,color='red')
        im =ax.plot(init[m,:],'--',label='initial for fwi',linewidth=6,color='black')
        plt.axis('tight')
        plt.legend(prop={'size': 20, 'weight':'bold'},loc='upper left')
    plt.savefig('./output/figure/inplabels',bbox_inches='tight')
    plt.show()
    plt.close()
    



def save_2drsf(model,par,path):
    
    binarf=path+'@'
    # save binary 
    model.astype('float32').tofile(binarf)
    cmd('''echo  ' # This is a Madagascar like header \n \n \t n1=%d \n \t n2=%d \n \t d1=%g \n 
    \t d2=%g \n \t o1=%g \n \t o2=%g \n \t data_format=native_float \n \t in=%s' > %s  
        '''%(par['nz'],par['nx'],par['dz'],par['dx'],par['oz'],par['ox'],binarf,path))


def cmd(command):
    """
    Run command and pipe what you would see in terminal into the output cell
    """
    print(command)
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stderr.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            # this prints the stdout in the end
            output2 = process.stdout.read().decode('utf-8')
            print(output2.strip())
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

