import numpy as np
import pandas as pd
import os
import torch
from numpy import random
from aspire.utils import coor_trans,Rotation
from aspire.volume import Volume
import aspire


def generateBallVoxel(center,radius,L):
    
    grid = coor_trans.grid_3d(L)
    voxel = ((grid['x']-center[0])**2 + (grid['y']-center[1])**2 + (grid['z']-center[2])**2) <= np.power(radius,2)
    
    return np.single(voxel.reshape((1,L**3)))
    

def generateCylinderVoxel(center,radius,L,axis = 2):
    
    grid = coor_trans.grid_3d(L)
    dims= ('x','y','z')
    cylinder_axes = tuple(dims[i] for i in range(3) if i != axis)
    voxel = ((grid[cylinder_axes[0]]-center[0])**2 + (grid[cylinder_axes[1]]-center[1])**2) <= np.power(radius,2)
    
    return np.single(voxel.reshape((1,L**3)))


def replicateVoxelSign(voxels):
    
    return Volume(np.concatenate((voxels.asnumpy(),-voxels.asnumpy()),axis=0))
    

def volsCovarEigenvec(vols,eigenval_threshold = 1e-3):
    vols_num = vols.shape[0]
    vols0mean = asnumpy((vols -  np.mean(vols,axis=0))).reshape((vols_num,-1))

    _,volsSTD,volsSpan = np.linalg.svd(vols0mean,full_matrices=False) 
    volsSTD /= np.sqrt(vols_num)  #standard devation is volsSTD / sqrt(n)
    eigenval_num = np.sum(volsSTD > np.sqrt(eigenval_threshold))
    volsSpan = volsSpan[:eigenval_num,:] * volsSTD[:eigenval_num,np.newaxis] 
    return Volume.from_vec(volsSpan) 


def rademacherDist(sz):
    val = random.randint(0,2,sz)
    val[val == 0] = -1
    return val

    
def cosineSimilarity(vec1,vec2):
    
    vec1 = asnumpy(vec1).reshape((vec1.shape[0],-1))
    vec2 = asnumpy(vec2).reshape((vec2.shape[0],-1))
    
    #vec1_norm = np.linalg.norm(vec1,axis=(-1)).reshape((vec1.shape[0],1))
    #vec2_norm = np.linalg.norm(vec2,axis=(-1)).reshape((vec2.shape[0],1))
    
    #vec1 = vec1/vec1_norm
    #vec2 = vec2/vec2_norm
    
    vec1 = np.linalg.svd(vec1,full_matrices=False)[2]
    vec2 = np.linalg.svd(vec2,full_matrices=False)[2]
    
    cosine_sim = np.matmul(vec1,vec2.transpose())
    
    
    
    return cosine_sim
    

def principalAngles(vec1,vec2):
    
    vec1 = asnumpy(vec1).reshape((vec1.shape[0],-1))
    vec2 = asnumpy(vec2).reshape((vec2.shape[0],-1))
    
    svd1 = np.linalg.svd(vec1,full_matrices=False)[2]
    svd2 = np.linalg.svd(vec2,full_matrices=False)[2]

    principal_angles = np.arccos(np.clip(np.abs(np.dot(svd1, svd2.T)), -1.0, 1.0))
    
    return np.min(np.degrees(principal_angles))

def asnumpy(data):
    if(type(data) == aspire.volume.volume.Volume or type(data) == aspire.image.image.Image):
        return data.asnumpy()
        
    return data


def np2torchDtype(np_dtype):

    return torch.float64 if (np_dtype == np.double) else torch.float32


def appendCSV(dataframe,csv_file):
    if(os.path.isfile(csv_file)):
        current_dataframe = pd.read_csv(csv_file,index_col =0)
        updated_dataframe = pd.concat([current_dataframe,dataframe],ignore_index = True)
        updated_dataframe.to_csv(csv_file)
        
    else:
        dataframe.to_csv(csv_file)
        
        
def meanCTFPSD(ctfs,L):
    ctfs_eval_grid = [np.power(ctf.evaluate_grid(L),2) for ctf in ctfs]
    return np.mean(np.array(ctfs_eval_grid),axis=0)
    