import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import torch
from numpy import random
from aspire.utils import coor_trans,Rotation,grid_2d,grid_3d
from aspire.volume import Volume
from aspire.source.image import ArrayImageSource
from aspire.storage.starfile import StarFile
from aspire.basis import FFBBasis3D
from aspire.reconstruction import MeanEstimator
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
    

def volsCovarEigenvec(vols,eigenval_threshold = 1e-3,randomized_alg = False,max_eigennum = None,weights = None):
    vols_num = vols.shape[0]
    if(weights is None): #If 
        vols_dist = np.ones(vols_num)/vols_num
    else:
        vols_dist = weights / np.sum(weights)
    vols_dist = vols_dist.astype(vols.dtype)
    vols_mean = np.sum(vols_dist[:, np.newaxis, np.newaxis, np.newaxis] * vols,axis=0)
    vols0mean = asnumpy((vols -  vols_mean)).reshape((vols_num,-1))

    if(not randomized_alg):
        vols0mean = np.sqrt(vols_dist[:, np.newaxis]) * vols0mean
        _,volsSTD,volsSpan = np.linalg.svd(vols0mean,full_matrices=False)
        #volsSTD /= np.sqrt(vols_num)  #standard devation is volsSTD / sqrt(n)
        eigenval_num = np.sum(volsSTD > np.sqrt(eigenval_threshold))
        volsSpan = volsSpan[:eigenval_num,:] * volsSTD[:eigenval_num,np.newaxis] 
    else:
        #TODO : add weights to randomized alg
        if(max_eigennum == None):
            max_eigennum = vols_num
        pca = PCA(n_components=max_eigennum,svd_solver='randomized')
        fitvols = pca.fit(vols0mean)
        volsSpan = fitvols.components_ * np.sqrt(fitvols.explained_variance_.reshape((-1,1)))

    return volsSpan


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

    vec1 = vec1.reshape((vec1.shape[0],-1))
    vec2 = vec2.reshape((vec2.shape[0],-1))
    vec1 = torch.linalg.svd(vec1,full_matrices = False)[2]
    vec2 = torch.linalg.svd(vec2,full_matrices = False)[2]
    cosine_sim = torch.matmul(vec1,torch.transpose(vec2,0,1).conj()).cpu().numpy()
    
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

dtype_mapping = {
    torch.float16 : torch.complex32,
    torch.complex32 : torch.float16,
    torch.float32 : torch.complex64,
    torch.complex64 : torch.float32,
    torch.float64 : torch.complex128,
    torch.complex128 : torch.float64
}
def get_complex_real_dtype(dtype):
    return dtype_mapping[dtype]


def get_torch_device():
    return torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")


def appendCSV(dataframe,csv_file):
    if(os.path.isfile(csv_file)):
        current_dataframe = pd.read_csv(csv_file,index_col =0)
        updated_dataframe = pd.concat([current_dataframe,dataframe],ignore_index = True)
        updated_dataframe.to_csv(csv_file)
        
    else:
        dataframe.to_csv(csv_file)
        
def soft_edged_kernel(radius,L,dim,radius_backoff = 2):
    #Implementation is based on RECOVAR https://github.com/ma-gilles/recovar/blob/main/recovar/mask.py#L106
    if(radius < 3):
        radius = 3
        print(f'Warning : radius value {radius} is too small. setting radius to 3 pixels.')
    if(dim == 2):
        grid_func = grid_2d
    elif(dim == 3):
        grid_func = grid_3d

    grid_radius = grid_func(L,shifted=True,normalized=False)['r']
    radius0 = radius - radius_backoff

    kernel = np.zeros(grid_radius.shape)

    kernel[grid_radius < radius0] = 1

    kernel = np.where((grid_radius >= radius0)*(grid_radius < radius),(1+np.cos(np.pi*(grid_radius-radius0)/(radius-radius0)))/2,kernel)

    return kernel / np.sum(kernel)


def meanCTFPSD(ctfs,L):
    ctfs_eval_grid = [np.power(ctf.evaluate_grid(L),2) for ctf in ctfs]
    return np.mean(np.array(ctfs_eval_grid),axis=0)
    
def sub_starfile(star_input,star_output,mrcs_index):
    star_out = StarFile(star_input)
    star_out['particles'] = pd.DataFrame(star_out['particles']).iloc[mrcs_index].to_dict(orient='list')
    star_out.write(star_output)

def estimateMean(source,basis = None):
    if(basis == None):
        L = source.L
        basis = FFBBasis3D((L,L,L))
    mean_estimator = MeanEstimator(source,basis = basis)
    mean_est = mean_estimator.estimate()

    return mean_est

def vol_fsc(vol1,vol2):
    if(type(vol1) != type(vol2)):
        raise Exception(f'Volumes of the same type expected vol1 is of type {type(vol1)} while vol2 is of type {type(vol2)}')

    if(type(vol1) == aspire.volume.Volume):
        return vol1.fsc(vol2)
    
    elif(type(vol1) == torch.Tensor):
        #TODO : implement faster FSC for torch tensors
        vol1 = Volume(vol1.cpu().numpy())
        vol2 = Volume(vol2.cpu().numpy())

        return vol1.fsc(vol2)

def relionReconstruct(inputfile,outputfile,classnum = None,overwrite = True,mrcs_index = None):
    if(mrcs_index is not None):
        subfile = f'{inputfile}.sub.tmp'
        sub_starfile(inputfile,subfile,mrcs_index)
        inputfile = subfile
    classnum_arg = f' --class {classnum}' if classnum is not None else ''
    inputfile_path,inputfile_name = os.path.split(inputfile)
    #outputfile_rel = os.path.relpath(outputfile,inputfile_path)
    outputfile_abs = os.path.abspath(outputfile)
    if(overwrite or (not os.path.isfile(outputfile))):
        os.system(f'cd {inputfile_path} && relion_reconstruct --i {inputfile_name} --o {outputfile_abs} --ctf' + classnum_arg)
        #compensate for volume sign inversion and normalization by image size in relion
        vol = (-1 * Volume.load(outputfile))
        vol*=vol.shape[-1]
        vol.save(outputfile,overwrite=True)
    else:
        vol = Volume.load(outputfile)
    if(mrcs_index is not None):
        os.remove(subfile)
    return vol


def readVols(volfiles):
    numvols = len(volfiles)
    vol_size = Volume.load(volfiles[0]).shape[-1]
    volumes = Volume(np.zeros((numvols,vol_size,vol_size,vol_size),dtype=np.float32))

    for i,volfile in enumerate(volfiles):
        volumes[i] = Volume.load(volfile)

    return volumes
    
