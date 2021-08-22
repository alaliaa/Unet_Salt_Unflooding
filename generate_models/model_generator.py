import random
import numpy as np
from  scipy.ndimage import gaussian_filter

def create_random_model(layer,nz):
    max_v = 4.3
    layer = (5 if layer<=4 else layer)
    vr       = np.zeros(nz)  
    vsm      = np.zeros(nz)
    lz       =  np.floor(np.random.rand(layer)*nz) # this return the depth for each layer
    lz       = np.sort(lz) # sort the depth
    lz[0] = 0
    vr[:]   = 1.5 + np.random.rand()*(max_v-1.5)   # This will initilize vel with upper bound vel of Max_v 
    # vr[:]   = 1500 + np.random.rand()*2700
    for i in range(layer):
        vr[int(lz[i-1]):int(lz[i])] = 1.5 + np.random.uniform(0.5,1)*(.5 + 3.0 * i/(layer-2))    # Here we define the velocity   
    if vr[vr > max_v].size !=0 :  vr[vr > max_v] = max_v # Setting an upper bound just in case, added by AA    
    vr[int(lz[layer-1]):] = 1.5 + np.random.uniform(0.8,1)*(max_v-1.5)
    water_depth = 20 + int(np.floor(np.random.rand()*15))
    vr[0:water_depth] = 1.5



    # smooth or not 
    if random.randint(0,1) or layer < 8: 
         vr = gaussian_filter(vr,sigma=np.random.uniform(2,10))
    # Always smooth 
    #vr[water_depth:] = gaussian_filter(vr[water_depth:],sigma=random2.uniform(4,6))

    
	

    # initial model for inversion 
    # vsm[water_depth:] = gaussian_filter(vr[water_depth::],sigma=20)
    # vsm[:water_depth] = 1.5
    #vsm[:] = vr [:]

    # set the salts
       # Vsalt = random.uniform(4.5,4.6) # aslt velocity
    Vsalt = 4.5
    salt_layers = random.randint(0,1)
    #salt_layers = random.randint(1,2)
    min_depth=int(np.floor(0.05*nz)) # arbitrary number, this minimum depth of salt is after the water_bottom
    if salt_layers == 1:
        salt_start = random.randint(int(water_depth+min_depth),nz-min_depth)
        salt_end = random.randint(salt_start+min_depth,nz)
        vr[salt_start:salt_end] = Vsalt 
    # elif salt_layers ==2: 
    #     # first salt
    #     salt_start1 = random.randint(int(water_depth+min_depth),nz-min_depth)
    #     salt_end1 = random.randint(salt_start1+min_depth,nz)
    #     #second salt
    #     salt_start2 = random.randint(int(water_depth+min_depth),nz-min_depth)
    #     salt_end2 = random.randint(salt_start2+min_depth,nz)
        
    #     if abs(salt_start2 - salt_end1) < 0.10*nz:
    #         vr[salt_start1:salt_end2] = Vsalt
    #     elif  abs(salt_start1 - salt_end2) < 0.10*nz : 
    #         vr[salt_start2:salt_end1] = Vsalt
    #     else: 
    #         vr[salt_start1:salt_end1] = Vsalt
    #         vr[salt_start2:salt_end2] = Vsalt

    # Flood
    vsm[:] = vr[:]
    top_salt = np.where(vr == 4.5)
    if len(top_salt)>0 and len(top_salt[0]>0) : vsm[top_salt[0][0]:] = Vsalt  
	

    return vr,vsm,water_depth




def create_random_model2(layer,nz,vel):
    max_v = 4.3
    layer = (5 if layer<=4 else layer)
    # vr       = np.zeros(nz)  
    vsm      = np.zeros(nz)
    lz       =  np.floor(np.random.rand(layer)*nz) # this return the depth for each layer
    lz       = np.sort(lz) # sort the depth
    lz[0] = 0
    vr = vel.copy()
    for i in range(layer):
        # Here we define the velocity   
        # The average velocity in the range is used to get a layery structure 
        vr[int(lz[i-1]):int(lz[i])] = np.average( np.random.normal(vr[int(lz[i-1]):int(lz[i])],0.6))    
        # vr[int(lz[i-1]):int(lz[i])] =  np.random.uniform(0.5,1) * np.average(vr[int(lz[i-1]):int(lz[i])]) 
    vr[int(lz[layer-1]):] = np.average(np.random.normal(vr[int(lz[layer-1]):],0.5)*(max_v)) # last layer !!
    # vr[int(lz[layer-1]):] = np.random.uniform(0.8,1) * np.average(vr[int(lz[layer-1]):]) 

    water_depth = 20 + int(np.floor(np.random.rand()*15))
    if vr[vr > max_v].size !=0 :  vr[vr > max_v] = max_v # Setting an upper bound just in case, added by AA    
    if vr[vr < 1.5].size !=0 :  vr[vr < 1.5] = 1.5 # Setting a lower bound just in case, added by AA 
    vr[0:water_depth] = 1.5



    # smooth or not 
    if random.randint(0,1) or layer < 8: 
         vr = gaussian_filter(vr,sigma=np.random.uniform(2,10))
    # Always smooth 
    # vr[water_depth:] = gaussian_filter(vr[water_depth:],sigma=np.random.uniform(2,10))

    
	

    # initial model for inversion 
    # vsm[water_depth:] = gaussian_filter(vr[water_depth::],sigma=20)
    # vsm[:water_depth] = 1.5
    #vsm[:] = vr [:]

    # set the salts
       # Vsalt = random.uniform(4.5,4.6) # aslt velocity
    Vsalt = 4.5
    salt_layers = random.randint(0,1)
    #salt_layers = random.randint(1,2)
    min_depth=int(np.floor(0.05*nz)) # arbitrary number, this minimum depth of salt is after the water_bottom
    if salt_layers == 1:
        salt_start = random.randint(int(water_depth+min_depth),nz-min_depth)
        salt_end = random.randint(salt_start+min_depth,nz)
        vr[salt_start:salt_end] = Vsalt 
 

    # Flood
    vsm[:] = vr[:]
    top_salt = np.where(vr == 4.5)
    if len(top_salt)>0 and len(top_salt[0]>0) : vsm[top_salt[0][0]:] = Vsalt  
	
    return vr,vsm,water_depth



# -=========================================================================

