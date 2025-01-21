import numpy as np
import torch
from os import path
import pickle
from matplotlib import pyplot as plt
import aspire
from umap import UMAP
from sklearn.metrics import auc
import click
from utils import *
from covar_sgd import CovarDataset,Covar,trainCovar
from covar_distributed import trainParallel
from wiener_coords import latentMAP,mahalanobis_threshold


def reconstructClass(starfile_path,vol_path,overwrite = False):
    '''
        Uses relion_reconstruct on each class in a star file
        if vol_path is a directory - generates a file per volume in the directory
        if vol_path is a file - combines all volumes into the file
    '''
    starfile = aspire.storage.StarFile(starfile_path)
    classes = np.unique(starfile['particles']['_rlnClassNumber'])
    classes = classes[np.where(classes.astype(np.float32)!=-1)] #unindentified images are labeled with class = -1 
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
        os.remove('vol_tmp.mrc')
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

def covar_workflow(starfile,rank,class_vols = None,whiten=True,noise_estimator = 'anisotropic',mask='fuzzy',generate_figs = True,save_data = True,skip_processing=False,debug = False,**training_kwargs):
    #Load starfile
    data_dir = os.path.split(starfile)[0]
    result_dir = path.join(data_dir,'result_data')
    dataset_path = os.path.join(result_dir,'dataset.pkl')
    #Only perform this when debug flag is False and there is no dataset pickle file already saved (In order to skip preprocessing when running multiple times for debugging)
    if((not debug) or (not os.path.isfile(dataset_path))): 
        if(not path.isdir(result_dir)):
            os.mkdir(result_dir)
        pixel_size = float(aspire.storage.StarFile(starfile)['optics']['_rlnImagePixelSize'][0])
        source = aspire.source.RelionSource(starfile,pixel_size=pixel_size)
        L = source.L
        
        mean_est = relionReconstruct(starfile,path.join(result_dir,'mean_est.mrc'),overwrite = False)

        if(class_vols is None):
            #Estimate ground truth states #TODO : should only be done if ground truth labels exist and some flag is given
            class_vols = reconstructClass(starfile,path.join(result_dir,'class_vols.mrc'))
        elif(class_vols.resolution != L): #Downsample ground truth volumes
            class_vols = class_vols.downsample(L)
        #Compute ground truth eigenvectors
        _,counts = np.unique(source.states[np.where(source.states.astype(np.float32)!=-1)],return_counts=True)
        states_dist = counts/np.sum(counts)
        covar_eigenvecs_gd = volsCovarEigenvec(class_vols,weights = states_dist)[:rank]

        #Print useful info TODO: use log files
        print(f'Norm squared of mean volume : {np.linalg.norm(mean_est)**2}')
        print(f'Top eigen values of ground truth covariance {np.linalg.norm(covar_eigenvecs_gd,axis=1)**2}')
        print(f'Correlation between mean volume and eigenvolumes {cosineSimilarity(torch.tensor(mean_est.asnumpy()),torch.tensor(covar_eigenvecs_gd))}')


        noise_est_num_ims = min(source.n,2**12)
        if(whiten): #TODO : speed this up without aspire implementation. for now noise estimation uses only 2**12 images
            if(noise_estimator == 'anisotropic'):
                noise_estimator = aspire.noise.AnisotropicNoiseEstimator(source[:noise_est_num_ims])
            elif(noise_estimator == 'white'):
                noise_estimator = aspire.noise.WhiteNoiseEstimator(source[:noise_est_num_ims])
            source = source.whiten(noise_estimator)
            noise_var = 1
        else:
            noise_estimator = aspire.noise.WhiteNoiseEstimator(source[:noise_est_num_ims])
            noise_var = noise_estimator.estimate()
        #TODO : if whiten is False normalize background will still normalize to get noise_var = 1 but this will not be taken into account - handle this.
        source = source.normalize_background(do_ramp=False)

        dataset = CovarDataset(source,noise_var,vectorsGD=covar_eigenvecs_gd,mean_volume=mean_est,mask=mask)
        dataset.states = torch.tensor(source.states) #TODO : do this at dataset constructor
        dataset.starfile = starfile
        dataset.class_vols = class_vols
        
        if(debug):
            with open(dataset_path,'wb') as fid:
                pickle.dump(dataset,fid)
    else:
        print(f"Reading pickled dataset from {dataset_path}")
        with open(dataset_path,'rb') as fid:
            dataset = pickle.load(fid)
        print(f"Dataset loaded successfuly")

    return covar_processing(dataset,rank,result_dir,generate_figs,save_data,skip_processing,**training_kwargs)

    

def covar_processing(dataset,covar_rank,result_dir,generate_figs = True,save_data = True,skip_processing = False,**training_kwargs):
    L = dataset.images.shape[-1]

    if(not skip_processing):
        #Perform optimization for eigenvectors estimation
        default_training_kwargs = {'batch_size' : 1024, 'max_epochs' : 50,
                                'lr' : 1e0,'optim_type' : 'Adam', #TODO : refine learning rate and reg values
                                'reg' : 1,'gamma_lr' : 0.8, 'gamma_reg' : 1,
                                'orthogonal_projection' : True,'nufft_disc' : None,
                                'num_reg_update_iters' : 3, 'use_halfsets' : False}
        #TODO : change batch_size into batch per GPU? 
        #TODO : change upsampling_factor into a training argument and pass that into Covar's methods instead of at constructor
        if('fourier_upsampling' in training_kwargs.keys()):
            upsampling_factor = training_kwargs['fourier_upsampling']
            del training_kwargs['fourier_upsampling']
        else:
            upsampling_factor = 2        
        default_training_kwargs.update(training_kwargs)

        optimize_in_fourier_domain = default_training_kwargs['nufft_disc'] is not None
        cov = Covar(L,covar_rank,pixel_var_estimate=dataset.signal_var,
                    fourier_domain=optimize_in_fourier_domain,upsampling_factor=upsampling_factor)
            
        if(torch.cuda.device_count() > 1): #TODO : implement halfsets for parallel training
            trainParallel(cov,dataset,savepath = path.join(result_dir,'training_results.bin'),
                **default_training_kwargs)
        else:
            cov = cov.to(get_torch_device())
            trainCovar(cov,dataset,savepath = path.join(result_dir,'training_results.bin'),
                **default_training_kwargs)
        
        
        #Compute wiener coordinates using estimated and ground truth eigenvectors
        eigen_est,eigenval_est= cov.eigenvecs
        eigen_est = eigen_est.to('cuda:0')
        eigenval_est = eigenval_est.to('cuda:0')
        coords_est,coords_covar_est = latentMAP(dataset,eigen_est,eigenval_est,return_coords_covar=True)

        eigenvals_GD = torch.norm(dataset.vectorsGD,dim=1) ** 2
        eigenvectors_GD = (dataset.vectorsGD / torch.sqrt(eigenvals_GD).unsqueeze(1)).reshape((-1,L,L,L))
        coords_GD,coords_covar_GD = latentMAP(dataset,eigenvectors_GD.to('cuda:0'),eigenvals_GD.to('cuda:0'),return_coords_covar=True)
        
        print(f'Eigenvalues of estimated covariance {eigenval_est}')

        num_states = torch.sum(torch.unique(dataset.states) != -1)
        cluster_centers = torch.zeros((num_states,covar_rank))
        class_vols_est = aspire.volume.Volume(np.zeros((num_states,L,L,L),dtype=np.float32))
        vol_tmp_file = 'vol_tmp.mrc'
        state_ind = 0
        for state in torch.unique(dataset.states):
            if(state != -1):
                mean_state_coord = torch.mean(coords_est[dataset.states == state],dim=0) #This uses the actual labels to compute the cluster center, used as a metric for covar reconstruction and not for the actual clustering.
                cluster_center_ind = torch.argmin(torch.norm(coords_est-mean_state_coord,dim=1))
                cluster_center = coords_est[cluster_center_ind]
                cluster_centers[state_ind] = cluster_center.cpu()
                state_coord_covar = coords_covar_est[cluster_center_ind]
                #state_coord_covar = torch.mean(coords_covar_est[dataset.states == state],dim=0) #TODO : is this the right way to compute this?
                index_under_threshold = mahalanobis_threshold(coords_est,cluster_center,state_coord_covar.to('cuda:0'))
                #index_under_threshold = mahalanobis_threshold(coords_est,cluster_center,state_coord_covar)
                print(f'Number of images used for reconstructing state {state} : {torch.sum(index_under_threshold)}')
                class_vols_est[state_ind] = relionReconstruct(dataset.starfile,vol_tmp_file,mrcs_index=index_under_threshold.cpu().numpy())
                state_ind += 1

        os.remove(vol_tmp_file)        
        class_vols_est.save(path.join(result_dir,'reconstructed_class_vols.mrc'),overwrite=True)

        class_vols_GD = dataset.class_vols

        v1 = class_vols_est.asnumpy().reshape(num_states,-1)
        v2 = class_vols_GD.asnumpy().reshape(num_states,-1)
        print(f'L2 norm error of class volumes {np.linalg.norm((v1-v2),axis=1)/np.linalg.norm(v2,axis=1)}')


        reducer = UMAP(n_components=2)
        umap_est = reducer.fit_transform(coords_est.cpu())
        cluster_centers_umap = reducer.transform(cluster_centers.cpu())
        umap_gd = reducer.fit_transform(coords_GD.cpu())

        data_dict = {}
        if(save_data):
            data_dict = {'eigen_est' : eigen_est.cpu(), 'eigenval_est' : eigenval_est.cpu(),
                        'eigenvals_GD' : eigenvals_GD.cpu(),'eigenvectors_GD' : eigenvectors_GD.cpu(),
                        'coords_est' : coords_est.cpu(),'coords_GD' : coords_GD.cpu(),
                        'coords_covar_est' : coords_covar_est, 'coords_covar_GD' : coords_covar_GD,
                        'umap_est' : umap_est, 'umap_gd' : umap_gd}
            with open(path.join(result_dir,'recorded_data.pkl'),'wb') as fid:
                pickle.dump(data_dict,fid)
    else:
        with open(path.join(result_dir,'recorded_data.pkl'),'rb') as fid:
            data_dict = pickle.load(fid)
            for key in data_dict.keys(): #Load each variable contained in data_dict
                exec(f"{key} = data_dict['{key}']") #TODO : load each key dynmaically 

    training_data = torch.load(path.join(result_dir,'training_results.bin'))


    #Generate plots
    figure_dict = {}
    if(generate_figs):
        fig_dir = path.join(result_dir,'result_figures')
        if(not path.isdir(fig_dir)):
            os.mkdir(fig_dir)
        
        fig,ax = plt.subplots()
        ax.scatter(coords_est[:,0].cpu(),coords_est[:,1].cpu(),c = dataset.states,s=0.1)
        for i in range(num_states):
            ax.annotate(f'{i}',(cluster_centers[i,0],cluster_centers[i,1]),fontweight='bold')
        figure_dict['wiener_coords_est'] = fig
        
        fig,ax = plt.subplots()
        ax.scatter(coords_GD[:,0].cpu(),coords_GD[:,1].cpu(),c = dataset.states,s=0.1)
        figure_dict['wiener_coords_gd'] = fig
        
        fig,ax = plt.subplots()
        ax.scatter(umap_est[:,0],umap_est[:,1],c=dataset.states,s=0.1)
        for i in range(num_states):
            ax.annotate(f'{i}',(cluster_centers_umap[i,0],cluster_centers_umap[i,1]),fontweight='bold')
        figure_dict['umap_coords_est'] = fig

        fig,ax = plt.subplots()
        ax.scatter(umap_gd[:,0],umap_gd[:,1],c=dataset.states,s=0.1)
        figure_dict['umap_coords_gd'] = fig

        fig,ax = plt.subplots()
        fsc = vol_fsc(class_vols_est,class_vols_GD)
        ax.plot(fsc[1].T)
        ax.legend([f'Class {i}' for i in range(num_states)])
        figure_dict['reconstructed_class_vol_fsc'] = fig

        rec_fsc = fsc[1]
        fsc_auc = []
        for i in range(class_vols_GD.shape[0]):
            fsc_auc.append(auc(np.arange(L//2)/L,np.abs(rec_fsc[i]).reshape(-1)))
        data_dict['fsc_auc_mean'] = np.mean(fsc_auc)
        data_dict['fsc_auc_std'] = np.std(fsc_auc)
        

        fig,ax = plt.subplots()
        fsc = vol_fsc(Volume(eigen_est.cpu().numpy()),Volume(eigenvectors_GD.cpu().numpy()))
        ax.plot(fsc[1].T)
        ax.legend([f'Eigenvector {i}' for i in range(covar_rank)])
        figure_dict['eigenvec_fsc'] = fig


        fig,ax = plt.subplots()
        ax.plot(training_data['log_epoch_ind'],[np.diag(c) for c in training_data['log_cosine_sim']])
        ax.legend([f'Eigenvector {i}' for i in range(covar_rank)])
        figure_dict['training_cosine_sim'] = fig

        fig,ax = plt.subplots()
        ax.plot(training_data['log_epoch_ind'],training_data['log_fro_err'])
        figure_dict['frobenius_norm_err'] = fig

        for f_name,f in figure_dict.items():
            f.savefig(path.join(fig_dir,f'{f_name}.jpg'))

    return data_dict,figure_dict,training_data,default_training_kwargs


def workflow_click_decorator(func):
    @click.option('-s','--starfile',type=str, help='path to star file.')
    @click.option('-r','--rank',type=int, help='rank of covariance to be estimated.')
    @click.option('-w','--whiten',type=bool,default=True,help='whether to whiten the images before processing')
    @click.option('--noise-estimator',type=str,default = 'anisotropic',help='noise estimator (white/anisotropic) used to whiten the images')
    @click.option('--mask',type=str,default='fuzzy',help="Type of mask to be used on the dataset. Can be either 'fuzzy' or path to a volume file/ Defaults to 'fuzzy'")
    @click.option('--skip-processing',is_flag = True,default = False,help='whether to disable logging of run to comet')
    @click.option('--debug',is_flag = True,default = False, help = 'debugging mode')
    @click.option('--batch-size',type=int,help = 'training batch size')
    @click.option('--max-epochs',type=int,help = 'number of epochs to train')
    @click.option('--lr',type=float,help= 'training learning rate')
    @click.option('--reg',type=float,help='regularization scaling')
    @click.option('--gamma-lr',type=float,help = 'learning rate decay rate')
    @click.option('--orthogonal-projection',type=bool,default = True,help = "force orthogonality of eigen vectors while training (default True)")
    @click.option('--nufft-disc',type=click.Choice([None,'bilinear','nearest']),default=None,help="Discretisation of NUFFT computation")
    @click.option('--fourier-upsampling',type=int,default=None,help='Upsaming factor in fourier domain for Discretisation of NUFFT. Only used when --nufft-disc is provided (default 2)')
    @click.option('--num-reg-update-iters',type=int,default=3,help='Number of iterations to update regularization')
    @click.option('--use-halfsets',type=bool,default=False,help='Whether to split data into halfsets for regularization update')
    def wrapper(*args,**kwargs):
        kwargs = {k : v for k,v in kwargs.items() if v is not None}
        return func(*args,**kwargs)
    
    return wrapper

@click.command()
@workflow_click_decorator
def covar_workflow_cli(starfile,rank,whiten=True,noise_estimator = 'anisotropic',mask='fuzzy',skip_processing = False,debug = False,**training_kwargs):
    covar_workflow(starfile,rank,whiten=whiten,noise_estimator=noise_estimator,mask=mask,skip_processing = skip_processing,debug = debug,**training_kwargs)

if __name__ == "__main__":
    covar_workflow_cli()
