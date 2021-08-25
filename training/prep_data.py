'''  prepare data for pytorch '''

import torch
from torch.utils.data import DataLoader




class Dataset():
    def __init__(self,x,y):
        assert  x.shape[0]==y.shape[0], 'shape 0 should be equal, ' 
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).float().cuda()
        self.y_data = torch.from_numpy(y).float().cuda()
        
        
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
        
        
    def __len__(self):
        return self.len
    

    
