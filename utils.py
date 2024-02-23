import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import torch
from numpy import random
from aspire.utils import coor_trans,Rotation
from aspire.volume import Volume
from aspire.source.image import ArrayImageSource
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
    

def volsCovarEigenvec(vols,eigenval_threshold = 1e-3,randomized_alg = False,max_eigennum = None):
    vols_num = vols.shape[0]
    vols0mean = asnumpy((vols -  np.mean(vols,axis=0))).reshape((vols_num,-1))

    if(not randomized_alg):
        _,volsSTD,volsSpan = np.linalg.svd(vols0mean,full_matrices=False)
        volsSTD /= np.sqrt(vols_num)  #standard devation is volsSTD / sqrt(n)
        eigenval_num = np.sum(volsSTD > np.sqrt(eigenval_threshold))
        volsSpan = volsSpan[:eigenval_num,:] * volsSTD[:eigenval_num,np.newaxis] 
    else:
        if(max_eigennum == None):
            max_eigennum = vols_num
        pca = PCA(n_components=max_eigennum,svd_solver='randomized')
        fitvols = pca.fit(vols0mean)
        volsSpan = fitvols.components_ * np.sqrt(fitvols.explained_variance_.reshape((-1,1)))

    return Volume.from_vec(volsSpan) 


def sim2imgsrc(sim):
    im_src =  ArrayImageSource(sim.images[:],pd.DataFrame(sim.get_metadata()),sim.angles)
    im_src.filter_indices = sim.filter_indices
    im_src.unique_filters = sim.unique_filters
    return im_src

def rademacherDist(sz):
    val = random.randint(0,2,sz)
    val[val == 0] = -1
    return val


def nonNormalizedGS(vecs):
    vecnum = vecs.shape[0]
    ortho_vecs = torch.zeros(vecs.shape)
    ortho_vecs[0] = vecs[0]
    for i in range(1,vecnum):
        ortho_vecs[i] = vecs[i]
        for j in range(i):
            ortho_vecs[i] = ortho_vecs[i] - torch.sum(vecs[i]*ortho_vecs[j])/(torch.norm(ortho_vecs[j])**2)*ortho_vecs[j]

    return ortho_vecs

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

def frobeniusNorm(vecs):
    #returns the frobenius norm of a matrix given by its eigenvectors (multiplied by the corresponding sqrt(eigenval))
    vecs = asnumpy(vecs).reshape((vecs.shape[0],-1))
    vecs_inn_prod = np.matmul(vecs,vecs.transpose())
    return np.sqrt(np.sum(vecs_inn_prod ** 2))
    

def frobeniusNormDiff(vec1,vec2):
    #returns the frobenius norm of the diffrence of two matrices given by their eigenvectors (multiplied by the corresponding sqrt(eigenval))
    
    vec1 = asnumpy(vec1).reshape((vec1.shape[0],-1))
    vec2 = asnumpy(vec2).reshape((vec2.shape[0],-1))

    normdiff_squared = frobeniusNorm(vec1) ** 2 + frobeniusNorm(vec2) ** 2  - 2*np.sum(np.matmul(vec1,vec2.transpose()) **2)
    
    return np.sqrt(normdiff_squared)


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
    