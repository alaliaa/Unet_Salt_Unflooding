# %%
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
from prep_data import Dataset
import torch
from torch.utils.data import DataLoader
from unet import UNet
import torch.nn as  nn
from util import plot_history, Plot_model , load_2drsf_data, plot_prediction, save_2drsf
import m8r as sf




def infer(x,y,name='',x0=0):
	''' This is to inder the 1D models'''
	#x, scalerx = scale_data(x.T)
	#y, scalery = scale_data(y.T)

	data  =Dataset(x,y)

	with torch.no_grad():
		# x = data.x_data.view(data.x_data.shape[0],1,-1)
		x = data.x_data
		y = data.y_data.view(data.y_data.shape[0],1,-1)
		pred = model(x)

		x = x.cpu().numpy()
		init = x[:,1,:] * 4.5
		x = x[:,0,:]
		y = y.cpu().numpy()
		y = y[:,0,:]
		pred = pred.cpu().numpy()
		pred = pred[:,0,:]


	#	x = scalerx.inverse_transform(x.T).T 
	#	y = scalery.inverse_transform(Y.T).T
	#	pred = scalerx.inverse_transform(pred.T).T
		x = x*4.5 
		y= y*4.5
		pred = pred*4.5
		# plot_prediction (x,y,pred,init,1,3) # the plot save the figure
		plot_prediction (x,y,pred,init,5,5) # the plot save the figure
		plt.savefig('./output/figure/Prediction'+name+'.pdf')
		plt.close()


def infer2(x):
	''' This is to nfer 2D model'''
	#x, scalerx = scale_data(x.T)
	#y, scalery = scale_data(y.T)
	
	x = torch.tensor(x,dtype=torch.float32).cuda()

	with torch.no_grad():
		# x = x.view(x.shape[0],1,-1)
		pred = model(x)
		pred = pred.cpu().numpy()
		pred = pred[:,0,:]
		x = x.cpu().numpy()
		x = x[:,0,:]


	#	x = scalerx.inverse_transform(x.T).T 
	#	Y = scalery.inverse_transform(Y.T).T
	#	pred = scalerx.inverse_transform(pred.T).T
		pred = pred*4.5
		# x = x*4.5
	#	for i in range (X.shape[0]):
	#		X[i,:] = X[i,:] + x0[i]
	#	Y = Y + X
	#	pred = pred + X


	# vmin, vmax = np.percentile(X.T,[2,98])
	# plt.figure()
	# plt.imshow(np.concatenate((X.T,pred.T),axis=0),cmap='jet',vmin=vmin,vmax=vmax,extent=[0,13.76,0,3.76*2])
	# plt.axis('tight')
	# plt.savefig('./output/figure/BP_pred')
	# plt.colorbar()
	# plt.show()
	return pred



if __name__ == "__main__" : 
	path= './output/'
	
	# infile= './BP_inv2.rsf'	
	opfile='./BP_inv_unflood_mean'
	infile = './BP_inv'
	# opfile = 'BP_inv7Hz_5km_unfloodf'
	
	# netname = 'unet' 
	netname = 'unet4' 
	# netname = 'unet2' # EAGE abstract 
	feature = 16
	# load nn model 
	model = UNet(2,1,feature)
	model.load_state_dict(torch.load(path+'NNmodel/'+netname))
	model.eval().cuda()
	
	
	# load data (training or validation)
	'''  For  EAGE abstract follow the form "path+'NNmodel/xvalid2.npy" '''
	x= np.load(path+'NNmodel/xvalid_%s.npy'%netname)
	y =np.load(path+'NNmodel/yvalid_%s.npy'%netname) 
	#x0= np.load(path+'NNmodel/x0valid.npy')
	infer (x,y,'_valid')

	x= np.load(path+'NNmodel/xtrain_%s.npy'%netname) 
	y =np.load(path+'NNmodel/ytrain_%s.npy'%netname) 
	#x0= np.load(path+'NNmodel/x0train.npy')
	infer(x,y,'')
	
	# loss_train = np.load(path+'NNmodel/Training_loss%s.npy'%netname)
	# loss_valid = np.load(path+'NNmodel/Validation_loss%s.npy'%netname)
	
	#r2_train = np.load(path+'NNmodel/Training_R2.npy')
	#r2_valid = np.load(path+'NNmodel/Validation_R2.npy')
	
	# plot_history(loss_train,loss_valid,netname) 
	#plot_r2(r2_train,r2_valid,'') 
	# plt.close()
	#plt.show()
	
	 # BP model 
	bp,par = load_2drsf_data(infile+'.rsf')
	bpf,_= load_2drsf_data('./BP_f.rsf')
	bp = bp.T
	bpf = bpf.T
	bp2 = bp.reshape((bp.shape[0],1,bp.shape[1]))
	bpf = bpf.reshape((bpf.shape[0],1,bpf.shape[1]))
	bp2 = np.concatenate((bp2,bpf),axis=1)

	bp2 =bp2/4.5
	bp2 = resize(bp2,(bp2.shape[0],bp2.shape[1],200))
	
	print("shape of the data is ", bp.shape)
	bp_corrected = infer2(bp2)
	#bp_corrected = bp_corrected.T
	bp_corrected = resize(bp_corrected,(par['nx'],par['nz']))
	Plot_model(bp.T,par,infile)
	Plot_model(bp_corrected.T,par,opfile)
	# Plot_model(bp.T,par)
	save_2drsf(bp_corrected,par,opfile+'.rsf')

	

# %%
