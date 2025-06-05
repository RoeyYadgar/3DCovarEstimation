import numpy as np
import torch
from os import path
import pickle
from matplotlib import pyplot as plt
import aspire
from umap import UMAP
from sklearn.metrics import auc
import click
from cov3d.utils import *
from cov3d.covar_sgd import trainCovar
from cov3d.dataset import CovarDataset,GTData
from cov3d.covar import Covar,CovarFourier,Mean
from cov3d.poses import PoseModule,pose_cryoDRGN2APIRE
from cov3d.covar_distributed import trainParallel
from cov3d.wiener_coords import latentMAP

def determineMaxBatchSize(devices,L,rank,dtype):
    devices_memory = [torch.cuda.get_device_properties(d).total_memory for d in devices]
    mem_per_device = min(devices_memory)
    model_size = L**3*rank*2*dtype.itemsize*3 # factor of 2 for complex numbers. factor of 3 comes from additional fourier reg tensor & vectorsGT (if exists)

    mem_for_batch = mem_per_device - model_size
    maximal_batch_size_per_device = mem_for_batch // (L**2*6*dtype.itemsize) # factor of 6 comes from complex number of images, 3d fourier points, and CTF filter 

    return maximal_batch_size_per_device*len(devices)


def reconstructClass(starfile_path,vol_path,overwrite = False):
    '''
        Uses relion_reconstruct on each class in a star file
        if vol_path is a directory - generates a file per volume in the directory
        if vol_path is a file - combines all volumes into the file
    '''
    starfile = aspire.storage.StarFile(starfile_path)
    classes = np.unique(starfile['particles']['_rlnClassNumber'])
    if(len(classes) == 1):
        print("Warning : rlnClassNumber contains only one class")
        return None
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


def load_mask(mask,L):
    if(mask == 'fuzzy'):
        mask = aspire.volume.Volume(aspire.utils.fuzzy_mask((L,)*3,dtype=np.float32))
    elif(path.isfile(mask)):
        mask = aspire.volume.Volume.load(mask)
        if(mask.resolution > L):
            mask = mask.downsample(L)

        min_mask_val = mask.asnumpy().min()
        max_mask_val = mask.asnumpy().max()
        if(np.abs(min_mask_val) > 1e-3 or np.abs(max_mask_val - 1) > 1e-3):
            print(f'Warning : mask volume range is [{min_mask_val},{max_mask_val}]. Normalzing mask')
            mask = (mask - min_mask_val) / (max_mask_val - min_mask_val)

    return mask
        

def check_dataset_sign(volume,mask):
     return np.sum((volume*mask).asnumpy()) > 0

def covar_workflow(starfile,rank,output_dir=None,whiten=True,noise_estimator = 'anisotropic',mask='fuzzy',optimize_pose=False,class_vols = None,gt_pose = None,debug = False,**training_kwargs):
    #Load starfile
    data_dir = os.path.split(starfile)[0]
    if(output_dir is None):
        output_dir = path.join(data_dir,'result_data')
    dataset_path = os.path.join(output_dir,'dataset.pkl')
    #Only perform this when debug flag is False and there is no dataset pickle file already saved (In order to skip preprocessing when running multiple times for debugging)
    if((not debug) or (not os.path.isfile(dataset_path))): 
        if(not path.isdir(output_dir)):
            os.mkdir(output_dir)
        star = aspire.storage.StarFile(starfile)
        states_in_source = '_rlnClassNumber' in star['particles']
        pixel_size = float(star['optics']['_rlnImagePixelSize'][0])
        source = aspire.source.RelionSource(starfile,pixel_size=pixel_size)
        if('_rlnOriginXAngst' in star['particles'] and '_rlnOriginYAngst' in star['particles']):
            #Aspire source only parses _rlnOriginX/Y and not _rlnOriginX/YAngst
            source.offsets =  np.array([star['particles']['_rlnOriginXAngst'],star['particles']['_rlnOriginYAngst']]).astype(np.float32).T / pixel_size #Convert to pixels
        L = source.L
        
        mean_est = relionReconstruct(starfile,path.join(output_dir,'mean_est.mrc'),overwrite = True) #TODO: change overwrite to True

        if(class_vols is None and states_in_source): #If class_vols was not provided but source has GT states use them to reconstruct GT
            #Estimate ground truth states
            class_vols = reconstructClass(starfile,path.join(output_dir,'class_vols.mrc'))
        elif(class_vols is not None): 
            if(isinstance(class_vols,str)):
                class_vols = Volume.load(class_vols) if os.path.isfile(class_vols) else readVols(class_vols)
                class_vols *= L
            if(class_vols.resolution != L): #Downsample ground truth volumes
                class_vols = class_vols.downsample(L)

        if(class_vols is not None):
            #Compute ground truth eigenvectors
            mean_gt = np.mean(class_vols,axis=0)
            _,counts = np.unique(source.states[np.where(source.states.astype(np.float32)!=-1)],return_counts=True)
            states_dist = counts/np.sum(counts)
            covar_eigenvecs_gt = volsCovarEigenvec(class_vols,weights = states_dist)[:rank]
        else:
            covar_eigenvecs_gt = None
            mean_gt = None

        #Print useful info TODO: use log files
        print(f'Norm squared of mean volume : {np.linalg.norm(mean_est)**2}')
        if(covar_eigenvecs_gt is not None):
            print(f'Top eigen values of ground truth covariance {np.linalg.norm(covar_eigenvecs_gt,axis=1)**2}')
            print(f'Correlation between mean volume and eigenvolumes {cosineSimilarity(torch.tensor(mean_est.asnumpy()),torch.tensor(covar_eigenvecs_gt))}')


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

        mask = load_mask(mask,L)
        invert_data = not check_dataset_sign(mean_est,mask)
        if(invert_data):
            print('inverting dataset sign')
            (-1 * mean_est).save(path.join(output_dir,'mean_est.mrc'),overwrite=True) #Save inverest mean volume. No need to invert the tensor itself as Dataset constructor expects uninverted volume
            if(mean_gt is not None):
                mean_gt *= -1
        dataset = CovarDataset(source,noise_var,mean_volume=mean_est,mask=mask,invert_data = invert_data,
                               apply_preprocessing = not optimize_pose) #When pose is being optimized the pre-processing must be done in the training loop itself
        dataset.starfile = starfile

        if(gt_pose is not None):
            gt_pose = pickle.load(open(gt_pose,'rb'))
            gt_rots,gt_offsets = pose_cryoDRGN2APIRE(gt_pose,L)
        else:
            gt_rots = None
            gt_offsets = None
        gt_data = GTData(covar_eigenvecs_gt,mean_gt,gt_rots,gt_offsets)

        
        if(debug):
            with open(dataset_path,'wb') as fid:
                pickle.dump(dataset,fid)
            with open(os.path.join(output_dir,'gt_data.pkl'),'wb') as fid:
                pickle.dump(gt_data,fid)
    else:
        print(f"Reading pickled dataset from {dataset_path}")
        with open(dataset_path,'rb') as fid:
            dataset = pickle.load(fid)
        with open(os.path.join(output_dir,'gt_data.pkl'),'rb') as fid:
            gt_data = pickle.load(fid)
        mean_est = Volume.load(path.join(output_dir,'mean_est.mrc'))
        mask = load_mask(mask,mean_est.resolution)
        print(f"Dataset loaded successfuly")

    covar_precoessing_output = covar_processing(dataset,rank,output_dir,mean_est,mask,optimize_pose,gt_data=gt_data,**training_kwargs)
    torch.cuda.empty_cache()
    return covar_precoessing_output

    

def covar_processing(dataset,covar_rank,output_dir,mean_volume_est=None,mask=None,optimize_pose=False,gt_data=None,**training_kwargs):
    L = dataset.images.shape[-1]

    #Perform optimization for eigenvectors estimation
    default_training_kwargs = {'batch_size' : 1024, 'max_epochs' : 20,
                            'lr' : 1e-5,'optim_type' : 'Adam', #TODO : refine learning rate and reg values
                            'reg' : 1,
                            'orthogonal_projection' : False,'nufft_disc' : 'bilinear',
                            'num_reg_update_iters' : 1, 'use_halfsets' : True,'objective_func' : 'ml'}
    
    #TODO : change upsampling_factor & objective_func into a training argument and pass that into Covar's methods instead of at constructor
    if('fourier_upsampling' in training_kwargs.keys()):
        upsampling_factor = training_kwargs['fourier_upsampling']
        del training_kwargs['fourier_upsampling']
    else:
        upsampling_factor = 2
    default_training_kwargs.update(training_kwargs)

    optimize_in_fourier_domain = default_training_kwargs['nufft_disc'] != 'nufft'
    covar_cls = Covar
    cov = covar_cls(L,covar_rank,pixel_var_estimate=dataset.signal_var,
                fourier_domain=optimize_in_fourier_domain,upsampling_factor=upsampling_factor)
    if(optimize_pose):
        mean = Mean(torch.tensor(mean_volume_est.asnumpy()),L,
                    fourier_domain=optimize_in_fourier_domain,
                    volume_mask=torch.tensor(mask.asnumpy()),
                    upsampling_factor=upsampling_factor)
        pose = PoseModule(dataset.rot_vecs,dataset.offsets,L)
    else:
        mean = None
        pose = None
        
    if(torch.cuda.device_count() > 1): #TODO : implement halfsets for parallel training
        trainParallel(cov,dataset,savepath = path.join(output_dir,'training_results.bin'),
            mean_model=mean,pose=pose,optimize_pose=optimize_pose,
            gt_data=gt_data,**default_training_kwargs)
    else:
        cov = cov.to(get_torch_device())
        trainCovar(cov,dataset,savepath = path.join(output_dir,'training_results.bin'),
            mean_model=mean,pose=pose,optimize_pose=optimize_pose,
            gt_data=gt_data,**default_training_kwargs)
    
    if(optimize_pose):
        #Update dataset with estimated pose and apply preprocessing
        #TODO: output pose to file
        dataset.pts_rot = dataset.compute_pts_rot(pose.get_rotvecs().cpu())
        dataset.preprocess_from_modules(mean,pose)
    
    #Compute wiener coordinates using estimated and ground truth eigenvectors
    eigen_est,eigenval_est= cov.eigenvecs
    eigen_est = eigen_est.to('cuda:0')
    eigenval_est = eigenval_est.to('cuda:0')
    coords_est,coords_covar_inv_est = latentMAP(dataset,eigen_est,eigenval_est,return_coords_covar=True)

    is_gt_eigenvols = gt_data.eigenvecs is not None
    if(is_gt_eigenvols):
        eigenvals_GT = torch.norm(gt_data.eigenvecs,dim=1) ** 2
        eigenvectors_GT = (gt_data.eigenvecs / torch.sqrt(eigenvals_GT).unsqueeze(1)).reshape((-1,L,L,L))
        coords_GT,coords_covar_inv_GT = latentMAP(dataset,eigenvectors_GT.to('cuda:0'),eigenvals_GT.to('cuda:0'),return_coords_covar=True)
    
    print(f'Eigenvalues of estimated covariance {eigenval_est}')


    data_dict = {'eigen_est' : eigen_est.cpu().numpy(), 'eigenval_est' : eigenval_est.cpu().numpy(),
                'coords_est' : coords_est.cpu().numpy(), 'coords_covar_inv_est' : coords_covar_inv_est.numpy(),
                'starfile' : os.path.abspath(dataset.starfile), 'data_sign_inverted' : dataset.data_inverted}
    if(is_gt_eigenvols):
        data_dict = {**data_dict,
                        'eigenvals_GT' : eigenvals_GT.cpu().numpy(),
                        'eigenvectors_GT' : eigenvectors_GT.cpu().numpy(),
                        'coords_GT' : coords_GT.cpu().numpy(),
                        'coords_covar_inv_GT' : coords_covar_inv_GT.numpy(),
                        }
    
    with open(path.join(output_dir,'recorded_data.pkl'),'wb') as fid:
        pickle.dump(data_dict,fid)
    if(dataset.mask is not None):
        aspire.volume.Volume(dataset.mask.cpu().numpy()).save(path.join(output_dir,'used_mask.mrc'),overwrite=True)
    #TODO: save eigenvolumes as MRC

    training_data = torch.load(path.join(output_dir,'training_results.bin'))

    return data_dict,training_data,default_training_kwargs


def workflow_click_decorator(func):
    @click.option('-s','--starfile',type=str, help='path to star file.')
    @click.option('-r','--rank',type=int, help='rank of covariance to be estimated.')
    @click.option('-o','--output-dir',type=str,help='path to output directory. when not provided a `result_data` directory will be used with the same path as the provided starfile')
    @click.option('-w','--whiten',type=bool,default=True,help='whether to whiten the images before processing')
    @click.option('--noise-estimator',type=str,default = 'anisotropic',help='noise estimator (white/anisotropic) used to whiten the images')
    @click.option('--mask',type=str,default='fuzzy',help="Type of mask to be used on the dataset. Can be either 'fuzzy' or path to a volume file/ Defaults to 'fuzzy'")
    @click.option('--optimize-pose',is_flag=True,default=False,help = 'Whether to optimize over image pose')
    @click.option('--class-vols',type=str,default=None,help='Path to GT volumes directory. Used if provided to log eigen vectors error metrics while training.Additionally, GT embedding is computed and logged')
    @click.option('--gt-pose',type=str,default=None,help='Path to GT pkl pose file (cryoDRGN format). Used if provided to log pose error metrics while training')
    @click.option('--debug',is_flag = True,default = False, help = 'debugging mode')
    @click.option('--batch-size',type=int,help = 'training batch size')
    @click.option('--max-epochs',type=int,help = 'number of epochs to train')
    @click.option('--lr',type=float,help= 'training learning rate')
    @click.option('--reg',type=float,help='regularization scaling')
    @click.option('--gamma-lr',type=float,help = 'learning rate decay rate')
    @click.option('--orthogonal-projection',type=bool,help = "force orthogonality of eigen vectors while training (default True)")
    @click.option('--nufft-disc',type=click.Choice(['bilinear','nearest','nufft']),default='bilinear',help="Discretisation of NUFFT computation")
    @click.option('--fourier-upsampling',type=int,help='Upsaming factor in fourier domain for Discretisation of NUFFT. Only used when --nufft-disc is provided (default 2)')
    @click.option('--num-reg-update-iters',type=int,help='Number of iterations to update regularization')
    @click.option('--use-halfsets',type=bool,help='Whether to split data into halfsets for regularization update')
    @click.option('--objective-func',type=click.Choice(['ml','ls']),default='ml',help='Which objective function to opimize. Either ml (maximum liklihood) or ls (least squares)')
    def wrapper(*args,**kwargs):
        kwargs = {k : v for k,v in kwargs.items() if v is not None}
        return func(*args,**kwargs)
    
    return wrapper

@click.command()
@workflow_click_decorator
def covar_workflow_cli(**kwargs):
    covar_workflow(**kwargs)

if __name__ == "__main__":
    covar_workflow_cli()
