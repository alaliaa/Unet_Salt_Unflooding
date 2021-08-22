import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import deepwave
from  scipy.ndimage import gaussian_filter, gaussian_filter1d

class fwi():
    def __init__(self,nx,nz,dx,nt,dt,
                num_dim,num_shots,num_sources_per_shot,ds,os,
                num_receivers_per_shot,dr,orec,source_depth,receivers_depth,num_batches,acquisition):
       self.nx=nx
       self.nz=nz
       self.dx=dx
       self.nt=nt
       self.dt=dt
       self.num_dims=num_dim
       self.num_shots=num_shots
       self.num_batches=num_batches
       self.num_sources_per_shot=num_sources_per_shot
       self.num_receivers_per_shot = num_receivers_per_shot 
       self.ds= ds
       self.dr= dr 
       self.sz = source_depth
       self.rz = receivers_depth
       self.os = os
       self.orec = orec
       self.s_cor, self.r_cor =self.get_coordinate(acquisition)


    def get_coordinate(self,mode):
       ''' Create arrays containing the source and receiver locations
        x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
        x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
        Note: the depth is set to zero , to change it change the first element in the last dimensino '''
       if self.num_shots ==1 and self.num_sources_per_shot==1 : # This will make the shot at the center !!
             x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
             x_s[:, 0, 1] = self.nx//2 * self.dx
             x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
             x_r[0, :, 1] = torch.arange(self.num_receivers_per_shot).float() * self.dr
             x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, self.num_sources_per_shot)


       else:
             x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
             # x direction 
            #  x_s[:,0,1] = torch.arange(self.os,self.os+self.num_shots * self.ds, self.ds).float() 
             x_s[:,0,1] = torch.tensor([0,7])  # this is for the training end-on spread three shots  
            #  x_s[:,0,1] = torch.tensor([0,4,7])  # this is for the training end-on spread three shots  
             # z direction  
             x_s[:, 0, 0] = self.sz
             if mode ==1:
               x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
               # x direction 
               x_r[0, :, 1] = torch.arange(self.orec,self.orec+self.num_receivers_per_shot* self.dr,self.dr).float() 
               x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
               # z direction 
               x_r[:, :, 0] = self.rz
             elif mode ==2: # fixed spread !! 
               x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
               # x direction
               # orec = [0,4,7]  ## for training 
               orec = [0,7]  ## for training 
               for i in range (self.num_shots):
                  # orec =  i * self.
                  # x_r[i, :, 1] = torch.arange(orec,orec+self.num_receivers_per_shot* self.dr,self.dr).float() 
                  x_r[i, :, 1] = torch.arange(orec[i],orec[i]+self.num_receivers_per_shot* self.dr,self.dr).float()  # for training 

               # z direction 
               x_r[:, :, 0] = self.rz

 
       return x_s,x_r


    def Ricker(self,freq):
        wavelet = (deepwave.wavelets.ricker(freq, self.nt, self.dt, 1/freq)
                                 .reshape(-1, 1, 1))

                        
        return wavelet

    def forward_modelling(self,model,wavelet,device):
       # pml_width parameter control the boundry, for free surface first argument should be 0 
       prop = deepwave.scalar.Propagator({'vp': model.to(device)}, self.dx,pml_width=[0,20,20,20,20,20])
#                                           survey_pad=[None, None, 200, 200])
       data = prop(wavelet.to(device), self.s_cor.to(device), self.r_cor.to(device), self.dt).cpu()
       return data
    
    


    def run_inversion(self,model,data_t,wavelet,msk,niter,alphatv,device): 
       ''' This run the full inverstion, check that all variable should be in gpu '''
       model = model.to(device)
       wavelet = wavelet.to(device)
       msk = torch.from_numpy(msk).int().to(device)
       model.requires_grad=True 
       m_max = model.max()
       m_min = model.min()
       data_t_max,_ = data_t.max(dim=0,keepdim=True)
      #  data_t_norm = data_t / (data_t_max.abs() + 1e-10)
 
       criterion = torch.nn.MSELoss()
       LR = 0.01

       optimizer = torch.optim.Adam([{'params':[model],'lr':LR}])

       num_batches = self.num_batches
       num_shots_per_batch = int(self.num_shots / num_batches)
       prop = deepwave.scalar.Propagator({'vp': model}, self.dx,pml_width=[0,20,20,20,20,20])
#                                           survey_pad=[None, None, 0, 0])
       t_start = time.time()
       loss_iter=[]
       increase = 0
       min_loss = 0
       first = True
       tol = 1e-4
   
       # updates is the output file
       updates=[]
       # main inversion loop 
       for itr in range(niter):
           running_loss = 0 
           running_loss1 =0 
           running_loss2 = 0
           optimizer.zero_grad()
	   # To compute TV I am cloning and detaching the model so we can apply weights(alphaTV) on it later 
           mTV = model.detach().clone()
           mTV.requires_grad=True
           optTV = torch.optim.Adam([{'params':[mTV],'lr':LR}]) # maybe Adam not the best choice here !! 

           for it in range(num_batches): # loop over shots 

               batch_wavl = wavelet.repeat(1, num_shots_per_batch, 1)
               batch_data_t = data_t[:,it::num_batches].to(device)
               batch_x_s = self.s_cor[it::num_batches].to(device)
               batch_x_r = self.r_cor[it::num_batches].to(device)
               batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)

               loss1 = criterion(batch_data_pred, batch_data_t)
               if loss1.item() == 0.0: # in case the initial and the true are the same (when we do not have salts)
                  updates.append(model.detach().cpu().numpy())
                  return np.array(updates)
               loss2 = tv_loss(mTV)
               # loss.backward()
               loss1.backward() # update L2 loss
               loss2.backward() # update TV loss
               # running_loss += loss.item()# + loss2.item()  
               running_loss += loss1.item() 
               # running_loss2 += loss2.item()

               # vmin, vmax = np.percentile(batch_data_t[:,2].cpu().numpy(), [2,98])
               # plt.figure()
               # plt.imshow(batch_data_t[:,2].cpu().numpy()-batch_data_pred[:,2].detach().cpu().numpy(), 
               # aspect='auto', vmin=vmin, vmax=vmax,cmap='gray')
                               
             
	   # smooth and normalize the gradient
           model.grad = self.grad_smooth(model,2,0.5).to(device)  
           model.grad =  self.grad_reg(model,mask=msk)

           mTV.grad = self.grad_reg(mTV,msk)
           # combine the two gradient     
           model.grad = model.grad + alphatv * mTV.grad
           model.grad =  self.grad_reg(model,mask=msk)


         #   if itr%5 ==0 :
         #    plt.figure(figsize=(10,5))
         #    plt.imshow(model.detach().cpu().numpy(),cmap='seismic')
         #    plt.colorbar()
         #    plt.show()     
         #    plt.figure(figsize=(10,5))
         #    plt.imshow(model.grad.cpu().numpy(),cmap='seismic')
         #    plt.colorbar()
         #    plt.show()            
         #    plt.figure(figsize=(10,5))
         #    plt.imshow(mTV.grad.cpu().numpy(),cmap='seismic')
         #    plt.colorbar()
         #    plt.show()
   
           optimizer.step()   
           model.data[model.data < m_min] = m_min
           model.data[model.data > m_max] = m_max
           loss_iter.append(running_loss)	
           print('Iteration: ', itr, 'Objective: ', running_loss)
         #   print('Running loss1: ',running_loss1,'running loss2: ',running_loss2)
           # see the gradient
         #   if itr%5 ==0 :
         #    plt.figure(figsize=(10,3))
         #    plt.imshow(model.clone().detach().cpu().numpy(),cmap='jet')
         #    plt.show()
         #    plt.figure(figsize=(10,3))
            # plt.imshow(model.grad.cpu().numpy(),cmap='seismic')
            # plt.colorbar()
            # plt.show()



###check convergence by avoiding increasing too much to avoid unecessery computation 
           if itr > 50 and loss_iter[itr-1]<loss_iter[itr]:
                  increase+=1
                  if first:
                     min_loss = loss_iter[itr-1]
                     model_min = model_old
                  first = False
           elif min_loss < loss_iter[itr] and not first: increase += 1
           elif first and np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>50:
               t_end = time.time()
               print('Runtime:',t_end-t_start)
               updates.append(model_min.cpu().numpy())
               return np.array(updates)
           elif  min_loss > loss_iter[itr] or min_loss==0:
                increase=0
                first = True
                model_min = model.detach().clone().cpu()
           # exit if the model did not decrease and return the best model :: later include a toleance        
           if increase ==10:
               t_end = time.time()
               print('Runtime:',t_end-t_start)
               updates.append(model_min.numpy())
               return np.array(updates)

           model_old = model.detach().clone().cpu()

       t_end = time.time()
       print('Runtime:',t_end-t_start)
       updates.append(model_min.numpy())
       return np.array(updates)



    def grad_smooth(self,model,sigmax,sigmaz):
               m =  model.detach().clone().cpu()
               gradient = model.grad.cpu()               
               gradient = gradient.sum(dim=1).view(gradient.shape[0],1).expand(-1,gradient.shape[1])
               gradient = gradient.numpy()
               # gradient = gaussian_filter1d(gradient,sigma=sigmaz,axis=0) # z
               gradient *= m.numpy()**3
               # gradient = gaussian_filter1d(gradient,sigma=sigmax,axis=1) # x 
               gradient = torch.tensor(gradient)
               return gradient


    def grad_reg(self,model,mask):
               # m =  model.detach().clone().cpu()
                       
               gradient = model.grad
               
               gmax     = (torch.abs(gradient)).max() 
               gradient = gradient / gmax  # normalize the gradient 
               gradient[0:mask,:] = 0 ## for training 
               # gradient = gradient * mask 

               return gradient
            




def tv_loss(model):
        h, w = model.shape 
        a = torch.pow(model[:h - 1, :w - 1] - model[1:, :w - 1],2)
        b = torch.pow(model[:h - 1, :w - 1] - model[:h - 1, 1:],2)
      #   penalize horizontal variations more than vertical
        return torch.sum(torch.pow(b +  a + 1e-15, 0.5))/(h*w)
      
