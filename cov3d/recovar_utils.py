import numpy as np
import os
import pickle
import recovar
from recovar import dataset as recovar_ds
from recovar import output as recovar_output
from cryodrgn.source import ImageSource
import torch
from scipy.ndimage import binary_dilation


def getRecovarDataset(starfile,split = True,perm = None,uninvert_data = False):
    #TODO: handle ctf and poses pkl files not in the same dir as star and mrcs files

    starfile_dir,starfile_name = os.path.split(starfile)    
    dataset_dict = {'datadir' : None,'uninvert_data' : uninvert_data}
    dataset_dict['ctf_file'] = os.path.join(starfile_dir,'ctf.pkl')
    dataset_dict['poses_file'] = os.path.join(starfile_dir,'poses.pkl')
    dataset_dict['particles_file'] = starfile
    
    if(split):
        num_ims = ImageSource.from_file(dataset_dict['particles_file']).n
        if(perm is None):
            perm = np.random.permutation(num_ims)
        ind_split = [perm[:num_ims//2],perm[num_ims//2:]]
        return recovar_ds.get_split_datasets_from_dict(dataset_dict, ind_split,lazy=True),perm
    else:
        return recovar_ds.load_dataset_from_dict(dataset_dict),None

def recovarReconstruct(inputfile,outputfile,overwrite = True,compute_mask=False):
     
    if(overwrite or (not os.path.isfile(outputfile))):
        dataset,_ = getRecovarDataset(inputfile)
        batch_size = recovar.utils.get_image_batch_size(dataset[0].grid_size, gpu_memory = recovar.utils.get_gpu_memory_total()) 
        noise_variance,_ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)
        mean = recovar.homogeneous.get_mean_conformation_relion(dataset,batch_size=batch_size,noise_variance = noise_variance,use_regularization=True)
        recovar_output.save_volume(mean[0]["combined"],outputfile.replace('.mrc',''),from_ft = True)

        if(compute_mask):
            volume_mask = recovar.mask.make_mask_from_half_maps_from_means_dict(mean[0], smax = 3 )
            kernel_size = 3
            dilation_iterations = np.ceil(6 * dataset[0].volume_shape[0] / 128).astype(int)
            dilated_volume_mask = binary_dilation(volume_mask,iterations=dilation_iterations)
            volume_mask = recovar.mask.soften_volume_mask(volume_mask, kernel_size)
            dilated_volume_mask = recovar.mask.soften_volume_mask(dilated_volume_mask, kernel_size)
            recovar_output.save_volume(dilated_volume_mask,outputfile.replace('.mrc','_mask'),from_ft = False)


    #return vol

def torch_to_numpy(arr):
    return arr.numpy() if isinstance(arr,torch.Tensor) else arr

def prepareDatasetForReconstruction(result_path):
    with open(result_path,'rb') as f:
        result = pickle.load(f)
    starfile = result['starfile']
    dataset,dataset_perm = getRecovarDataset(starfile,uninvert_data=result['data_sign_inverted'])
    batch_size = recovar.utils.get_image_batch_size(dataset[0].grid_size, gpu_memory = recovar.utils.get_gpu_memory_total()) 
    noise_variance,_ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)

    zs = result['coords_est'][dataset_perm]
    cov_zs = result['coords_covar_inv_est'][dataset_perm]

    return dataset,zs,cov_zs,noise_variance,dataset_perm

def recovarReconstructFromEmbedding(inputfile,outputfolder,embedding_positions,n_bins=30):
    dataset,zs,cov_zs,noise_variance,dataset_perm = prepareDatasetForReconstruction(inputfile)
    L = dataset[0].grid_size
    B_factor = 0 #TODO: handle B_factor
    if(os.path.isfile(embedding_positions)):
        with open(embedding_positions,'rb') as f:
            embedding_positions = pickle.load(f)
    
    recovar_output.compute_and_save_reweighted(dataset, embedding_positions, zs, cov_zs, noise_variance*np.ones(L//2-1), outputfolder, B_factor, n_bins = n_bins)
