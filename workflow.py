import numpy as np
import torch
from os import path
import pickle
from matplotlib import pyplot as plt
import aspire
from utils import *
from covar_sgd import CovarDataset,Covar
from covar_distributed import trainParallel
from wiener_coords import wiener_coords


def reconstructClass(starfile_path,vol_path,overwrite = False):
    '''
        Uses relion_reconstruct on each class in a star file
        if vol_path is a directory - generates a file per volume in the directory
        if vol_path is a file - combines all volumes into the file
    '''
    starfile = aspire.storage.StarFile(starfile_path)
    classes = np.unique(starfile['particles']['_rlnClassNumber'])
    img_size = int(float(starfile['optics']['_rlnImageSize'][0]))
    
    is_vol_path_dir = path.isdir(vol_path)
    if(not is_vol_path_dir):
        vol = aspire.volume.Volume(np.zeros((len(classes),img_size,img_size,img_size),dtype=np.float32))
        if(overwrite or (not path.isfile(vol_path))):
            vol_file = 'vol_tmp.mrc'
            for i,c in enumerate(classes):
                vol[i] = relionReconstruct(starfile_path,vol_file,classnum=c)
                
        else:
            return Volume.load(vol_path)
        vol.save(vol_path,overwrite=True)
        return vol
    else:
        for i,c in enumerate(classes):
            vol_file = f'{vol_path}/vol_{c}.mrc'
            if(overwrite or (not path.isfile(vol_file))):
                relionReconstruct(starfile_path,vol_file,classnum=c)   

def normalizeRelionVolume(vol,source,batch_size = 512):
    image_volproj_product = 0
    volproj2_product = 0
    for i in range(0,source.n,batch_size):
        projected_vol = source.vol_forward(vol,i,batch_size).asnumpy()
        images = source.images[i:i+batch_size].asnumpy()
        image_volproj_product += np.sum(projected_vol*images)
        volproj2_product += np.sum(projected_vol**2)
        print(image_volproj_product/volproj2_product)

    scale_const = image_volproj_product/volproj2_product
    print(scale_const)
        
    return scale_const

def covar_workflow(starfile,covar_rank,covar_eigenvecs = None,whiten=True,noise_estimator = 'anisotropic',generate_figs = True,save_data = True):
    #Load starfile
    data_dir = os.path.split(starfile)[0]
    result_dir = path.join(data_dir,'result_data')
    if(not path.isdir(result_dir)):
        os.mkdir(result_dir)
    pixel_size = float(aspire.storage.StarFile(starfile)['optics']['_rlnImagePixelSize'][0])
    source = aspire.source.RelionSource(starfile,pixel_size=pixel_size)
    L = source.L

    
    if(whiten):
        #TODO : use normalize background as well
        if(noise_estimator == 'anisotropic'):
            noise_estimator = aspire.noise.AnisotropicNoiseEstimator(source)
        elif(noise_estimator == 'white'):
            noise_estimator = aspire.noise.WhiteNoiseEstimator(source)
        source = source.whiten(noise_estimator)
        noise_var = 1
    else:
        noise_estimator = aspire.noise.WhiteNoiseEstimator(source)
        noise_var = noise_estimator.estimate()
    
    mean_est = relionReconstruct(starfile,path.join(result_dir,'mean_est.mrc'),overwrite = False)

    if(covar_eigenvecs is None):
        #Estimate ground truth states:
        class_vols = reconstructClass(starfile,path.join(result_dir,'class_vols.mrc'))
        #Compute ground truth eigenvectors
        _,counts = np.unique(source.states,return_counts=True)
        states_dist = counts/np.sum(counts)
        covar_eigenvecs_gd = volsCovarEigenvec(class_vols,weights = states_dist)[:covar_rank]
    else:
        covar_eigenvecs_gd = covar_eigenvecs.asnumpy().reshape((covar_rank,-1))

    #Print useful info TODO: use log files
    print(f'Norm squared of mean volume : {np.linalg.norm(mean_est)**2}')
    print(f'Top eigen values of ground truth covariance {np.linalg.norm(covar_eigenvecs_gd,axis=1)**2}')
    print(f'Correlation between mean volume and eigenvolumes {cosineSimilarity(torch.tensor(mean_est.asnumpy()),torch.tensor(covar_eigenvecs_gd))}')

    #Perform optimization for eigenvectors estimation
    

    dataset = CovarDataset(source,noise_var,vectorsGD=covar_eigenvecs_gd,mean_volume=mean_est)
    cov = Covar(L,covar_rank,pixel_var_estimate=dataset.signal_var)
    trainParallel(cov,dataset,savepath = path.join(result_dir,'training_results.bin'),
                    batch_size = 32,
                    max_epochs = 10,
                    lr = 1e-2,optim_type = 'Adam',
                    reg = 1e-6,
                    gamma_lr = 0.8,
                    gamma_reg = 0.8,
                    orthogonal_projection= True)
    
    #Compute wiener coordinates using estimated and ground truth eigenvectors
    eigen_est,eigenval_est= cov.eigenvecs
    eigen_est = eigen_est.to('cuda:0')
    eigenval_est = eigenval_est.to('cuda:0')
    coords_est = wiener_coords(dataset,eigen_est,eigenval_est)

    eigenvals_GD = np.linalg.norm(covar_eigenvecs_gd,axis=1) ** 2
    eigenvectors_GD = (covar_eigenvecs_gd / np.sqrt(eigenvals_GD[:,np.newaxis])).reshape((-1,L,L,L))
    coords_GD = wiener_coords(dataset,torch.tensor(eigenvectors_GD).to('cuda:0'),torch.tensor(eigenvals_GD).to('cuda:0'),8)
    
    print(f'Eigenvalues of estimated covariance {eigenval_est}')

    #Generate plots
    if(generate_figs):
        fig_dir = path.join(result_dir,'result_figures')
        if(not path.isdir(fig_dir)):
            os.mkdir(fig_dir)
        f = plt.figure()
        plt.scatter(coords_est[:,0].cpu(),coords_est[:,1].cpu(),c = source.states)
        f.savefig(path.join(fig_dir,'wiener_coords_est.jpg'))

        f = plt.figure()
        plt.scatter(coords_GD[:,0].cpu(),coords_GD[:,1].cpu(),c = source.states)
        f.savefig(path.join(fig_dir,'wiener_coords_gd.jpg'))

        f = plt.figure()
        ax = plt.axes()
        fsc = vol_fsc(Volume(eigen_est.cpu().numpy()),Volume(eigenvectors_GD))
        ax.plot(fsc[1].T)
        ax.legend([f'Eigenvector {i}' for i in range(covar_rank)])
        ax.figure.savefig(path.join(fig_dir,'eigenvec_fsc.jpg'))

        f = plt.figure()
        ax = plt.axes()
        training_data = torch.load(path.join(result_dir,'training_results.bin'))
        ax.plot([np.diag(c) for c in training_data['log_cosine_sim']])
        ax.legend([f'Eigenvector {i}' for i in range(covar_rank)])
        ax.figure.savefig(path.join(fig_dir,'training_cosine_sim.jpg'))

    if(save_data):
        data_dict = {'eigen_est' : eigen_est, 'eigenval_est' : eigenval_est,
                     'eigenvals_GD' : eigenvals_GD,'eigenvectors_GD' : eigenvectors_GD,
                      'coords_est' : coords_est,'coords_GD' : coords_GD}
        with open(path.join(result_dir,'recorded_data.pkl'),'wb') as fid:
            pickle.dump(data_dict,fid)

if __name__ == "__main__":
    #covar_workflow('/scratch/roaiyadgar/data/cryoDRGN_dataset/uniform/particles.128.ctf_preprocessed_L64.star',covar_rank = 5,whiten=True)
    recovar_eigenvecs = aspire.volume.Volume.load('/scratch/roaiyadgar/data/empiar10076/result_data/recovar_eigenvecs.mrc')
    covar_workflow('/scratch/roaiyadgar/data/empiar10076/L17Combine_weight_local_preprocessed_L64.star',covar_rank = 5,covar_eigenvecs=recovar_eigenvecs,whiten=True)
