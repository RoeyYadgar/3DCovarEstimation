import numpy as np
import os
import pickle
import recovar
from recovar import dataset as recovar_ds
from recovar import output as recovar_output
import torch
#def aspireSource2RecovarDataset():
'''
def getRecovarDataset(source,starfile,ctf_file = None):
    #pixel_size
    starfile_dir = os.path.split(starfile)[0]
    #mrcs_file = os.path.join(starfile_dir,'starfilmrcs')
    mrcs_name = source.get_metadata()[0][0].split('@')[1] #Assuming a single mrcs file is used
    mrcs_file = os.path.join(starfile_dir,mrcs_name)
    image_stack = recovar_ds.MRCDataMod(mrcs_file)

    ctf_params = np.array(load_ctf_for_training(dataset.D, ctf_file))
    dataset = recovar_ds.CryoEMDataset( image_stack, source.pixel_size,
                            source.rotations, source.offsets, ctf_params[:,1:], CTF_fun = CTF_fun, dataset_indices = ind, tilt_series_flag = tilt_series)
'''

def getRecovarDataset(starfile,split = True,perm = None):
    
    starfile_dir,starfile_name = os.path.split(starfile)
    #dataset_dict = recovar_ds.get_default_dataset_option()
    dataset_dict = {'datadir' : None}
    dataset_dict['ctf_file'] = os.path.join(starfile_dir,'ctf.pkl')
    dataset_dict['poses_file'] = os.path.join(starfile_dir,'poses.pkl')
    dataset_dict['particles_file'] = os.path.join(starfile_dir,starfile_name.replace('.star','.mrcs')) #TODO: handle different file names between star and mrcs
    
    if(split):
        num_ims = len(recovar_ds.MRCDataMod(dataset_dict['particles_file']))
        if(perm is None):
            perm = np.random.permutation(num_ims)
        ind_split = [perm[:num_ims//2],perm[num_ims//2:]]
        return recovar_ds.get_split_datasets_from_dict(dataset_dict, ind_split),perm
    else:
        return recovar_ds.load_dataset_from_dict(dataset_dict),None

def recovarReconstruct(inputfile,outputfile,overwrite = True):
     
    if(overwrite or (not os.path.isfile(outputfile))):
        dataset = getRecovarDataset(inputfile)
        batch_size = recovar.utils.get_image_batch_size(dataset[0].grid_size, gpu_memory = recovar.utils.get_gpu_memory_total()) 
        noise_variance,_ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)
        mean = recovar.homogeneous.get_mean_conformation_relion(dataset,batch_size=batch_size,noise_variance = noise_variance,use_regularization=True)
        recovar_output.save_volume(mean[0]["combined"],outputfile.replace('.mrc',''),from_ft = True)

    #return vol

def torch_to_numpy(arr):
    return arr.numpy() if isinstance(arr,torch.Tensor) else arr


def recovarReconstructFromEmbedding(inputfile,outputfolder,cov_results_path,embedding_positions,n_bins=30):
    dataset,dataset_perm = getRecovarDataset(inputfile)
    batch_size = recovar.utils.get_image_batch_size(dataset[0].grid_size, gpu_memory = recovar.utils.get_gpu_memory_total()) 
    noise_variance,_ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)
    L = 64
    with open(cov_results_path,'rb') as f:
        cov_results = pickle.load(f)
        zs = cov_results['coords_est']
        cov_zs = cov_results['coords_covar_est']
        cov_zs = torch.linalg.inv(cov_zs) #For some reason recovar expects the inverse of the covariance matrix?
        #TODO : remove torch_to_numpy
        zs = torch_to_numpy(zs)
        cov_zs = torch_to_numpy(cov_zs)
        zs = zs[dataset_perm]
        cov_zs = cov_zs[dataset_perm]
    B_factor = 0 #TODO: handle B_factor
    with open(embedding_positions,'rb') as f:
        embedding_positions = pickle.load(f)
    recovar_output.compute_and_save_reweighted(dataset, embedding_positions, zs, cov_zs, noise_variance*np.ones(L//2-1), outputfolder, B_factor, n_bins = n_bins)

    

if __name__ == "__main__":
    #s = recovarReconstruct('/data/roaiyadgar/data/igg_1d/images/snr0.01/downsample_L128/snr0.01.star',outputfile = "exp_data/test.mrc")
    recovarReconstructFromEmbedding('/data/roaiyadgar/data/igg_1d/images/snr0.01/downsample_L64/snr0.01.star','exp_data','/data/roaiyadgar/data/igg_1d/images/snr0.01/downsample_L64/result_data/recorded_data.pkl','exp_data/embedding_positions.pkl')