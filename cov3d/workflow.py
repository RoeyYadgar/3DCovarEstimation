import numpy as np
import torch
from os import path
import pickle
import aspire
import click
from cov3d.utils import *
from cov3d.source import ImageSource
from cov3d.covar_sgd import trainCovar
from cov3d.dataset import CovarDataset,LazyCovarDataset,GTData
from cov3d.covar import Covar,CovarFourier,Mean
from cov3d.poses import PoseModule,pose_cryoDRGN2APIRE,pose_ASPIRE2cryoDRGN,out_of_plane_rot_error,offset_mean_error
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

def covar_workflow(inputfile,rank,output_dir=None,poses=None,ctf=None,lazy=False,whiten=True,mask='fuzzy',optimize_pose=False,optimize_contrast=False,class_vols = None,gt_pose = None,debug = False,gt_path = None,**training_kwargs):
    data_dir = os.path.split(inputfile)[0]
    if(output_dir is None):
        output_dir = path.join(data_dir,'result_data')
    dataset_path = os.path.join(output_dir,'dataset.pkl')
    #Only perform this when debug flag is False and there is no dataset pickle file already saved (In order to skip preprocessing when running multiple times for debugging)
    if((not debug) or (not os.path.isfile(dataset_path))): 
        if(not path.isdir(output_dir)):
            os.mkdir(output_dir)

        source = ImageSource(inputfile,poses_path=poses,ctf_path=ctf,apply_preprocessing=whiten)
        noise_var = source.estimate_noise_var() if not whiten else 1
        L = source.resolution
        
        mean_est = relionReconstruct(inputfile,path.join(output_dir,'mean_est.mrc'),overwrite = True) #TODO: change overwrite to True

        #Print useful info TODO: use log files
        print(f'Norm squared of mean volume : {np.linalg.norm(mean_est)**2}')


        mask = load_mask(mask,L)
        invert_data = not check_dataset_sign(mean_est,mask)
        if(invert_data):
            print('inverting dataset sign')
            mean_est = -1 * mean_est
            mean_est.save(path.join(output_dir,'mean_est.mrc'),overwrite=True) #Save inverest mean volume. No need to invert the tensor itself as Dataset constructor expects uninverted volume


        dataset_cls = CovarDataset if not lazy else LazyCovarDataset
        dataset = dataset_cls(source,noise_var,mean_volume=mean_est,mask=mask,invert_data = invert_data,
                               apply_preprocessing = not optimize_pose) #When pose is being optimized the pre-processing must be done in the training loop itself
        dataset.starfile = inputfile

        if gt_path is None:
            #Construct GT data from availble inputs
            if class_vols is not None: 
                class_labels = None
                if isinstance(class_vols,str):
                    assert os.path.isfile(class_vols), f'class_vols {class_vols} is not a file'
                    class_vols = Volume.load(class_vols)
                    #TODO: support pickle file with class labels

                elif isinstance(class_vols,list): #list of mrc files                
                    class_vols = readVols(class_vols)

                #class_vols *= L
                if class_vols.resolution != L: #Downsample ground truth volumes
                    class_vols = class_vols.downsample(L)
            else:
                class_labels = None
                if inputfile.endswith('.star'):
                    #Check if class labels are present in the starfile
                    star = aspire.storage.StarFile(inputfile)
                    if '_rlnClassNumber' in star['particles']:
                        class_labels = np.array([float(v) for v in star['particles']['_rlnClassNumber']])
                        class_vols = reconstructClass(inputfile,path.join(output_dir,'class_vols.mrc'))
                #TODO: support other file types

            if(class_vols is not None):
                #Compute ground truth eigenvectors
                mean_gt = np.mean(class_vols,axis=0)
                if invert_data:
                    mean_gt *= -1
                if class_labels is not None:
                    _,counts = np.unique(class_labels[np.where(class_labels!=-1)],return_counts=True)
                    states_dist = counts/np.sum(counts)
                else:
                    states_dist = None
                covar_eigenvecs_gt = volsCovarEigenvec(class_vols,weights = states_dist)[:rank]
            else:
                covar_eigenvecs_gt = None
                mean_gt = None

            if(covar_eigenvecs_gt is not None):
                print(f'Top eigen values of ground truth covariance {np.linalg.norm(covar_eigenvecs_gt,axis=1)**2}')
                print(f'Correlation between mean volume and eigenvolumes {cosineSimilarity(torch.tensor(mean_est.asnumpy()),torch.tensor(covar_eigenvecs_gt))}')

            if(gt_pose is not None):
                gt_pose = pickle.load(open(gt_pose,'rb'))
                gt_rots,gt_offsets = pose_cryoDRGN2APIRE(gt_pose,L)
            else:
                gt_rots = None
                gt_offsets = None
            gt_data = GTData(covar_eigenvecs_gt,mean_gt,gt_rots,gt_offsets)
        else:
            with open(gt_path,'rb') as fid:
                gt_data = pickle.load(fid)

        
        if(debug):
            with open(dataset_path,'wb') as fid:
                pickle.dump(dataset,fid)
            with open(os.path.join(output_dir,'gt_data.pkl'),'wb') as fid:
                pickle.dump(gt_data,fid)
    else:
        print(f"Reading pickled dataset from {dataset_path}")
        with open(dataset_path,'rb') as fid:
            dataset = pickle.load(fid)

        if gt_path is None: #If gt_path is not provided, load from output_dir
            gt_path = os.path.join(output_dir,'gt_data.pkl')
        with open(gt_path,'rb') as fid:
            gt_data = pickle.load(fid)
        mean_est = Volume.load(path.join(output_dir,'mean_est.mrc'))
        mask = load_mask(mask,mean_est.resolution)
        print(f"Dataset loaded successfuly")

    covar_precoessing_output = covar_processing(dataset,rank,output_dir,mean_est,mask,optimize_pose,optimize_contrast,gt_data=gt_data,**training_kwargs)
    torch.cuda.empty_cache()
    return covar_precoessing_output

    

def covar_processing(dataset,covar_rank,output_dir,mean_volume_est=None,mask=None,optimize_pose=False,optimize_contrast=False,gt_data=None,**training_kwargs):
    L = dataset.resolution

    #Perform optimization for eigenvectors estimation
    default_training_kwargs = {'batch_size' : 1024, 'max_epochs' : 20,
                            'lr' : 1e-6,'optim_type' : 'Adam', #TODO : refine learning rate and reg values
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
        pose = PoseModule(dataset.rot_vecs,dataset.offsets,L,use_contrast=optimize_contrast)
        init_pose = (torch.tensor(Rotation.from_rotvec(dataset.rot_vecs.numpy())),dataset.offsets.clone())
    else:
        mean = None
        pose = None
        
    if(torch.cuda.device_count() > 1): #TODO : implement halfsets for parallel training
        trainParallel(cov,dataset,savepath = path.join(output_dir,'training_results.bin'),
            mean_model=mean,pose=pose,optimize_pose=optimize_pose,
            gt_data=gt_data,**default_training_kwargs)

        #When using lazy dataset and DDP, this instance of the dataset is not the same as the one modified in trainParallel
        #therefore we need to call the setup method directly
        if isinstance(dataset,LazyCovarDataset):
            dataset.post_init_setup(fourier_domain=False)
    else:
        cov = cov.to(get_torch_device())
        trainCovar(cov,dataset,savepath = path.join(output_dir,'training_results.bin'),
            mean_model=mean,pose=pose,optimize_pose=optimize_pose,
            gt_data=gt_data,**default_training_kwargs)
    
    if(optimize_pose):
        pose = pose.to('cpu')
        #Print how signficat the refined pose was changed from the given initial pose
        rot_change = out_of_plane_rot_error(torch.tensor(Rotation.from_rotvec(pose.get_rotvecs().numpy())),
                                            init_pose[0])[1]
        print(f'Rotation out-of-plane change: {rot_change} degrees')
        offset_change = offset_mean_error(pose.get_offsets(),init_pose[1])
        print(f'Image offset change: {offset_change} pixels')
        
        #Update dataset with estimated pose and apply preprocessing
        dataset.update_pose(pose)
        dataset.preprocess_from_modules(mean,pose)

        #Dump refined pose and mean volume
        refined_pose = pose_ASPIRE2cryoDRGN(Rotation.from_rotvec(pose.get_rotvecs().cpu().numpy()).matrices,
                                            pose.get_offsets().cpu().numpy(),L)
        with open(path.join(output_dir,'refined_poses.pkl'),'wb') as fid:
            pickle.dump(refined_pose,fid)
        if(pose.use_contrast):
            with open(path.join(output_dir,'contrast.pkl'),'wb') as fid:
                pickle.dump(pose.get_contrasts(),fid)

        mean_volume_est = Volume(mean.get_volume_spatial_domain().detach().cpu().numpy())
    
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
                'mean_est' : mean_volume_est.asnumpy() if mean_volume_est is not None else None,
                'starfile' : os.path.abspath(dataset.starfile) if dataset.starfile is not None else None,
                'data_sign_inverted' : dataset.data_inverted}
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
    @click.option('-i','--inputfile',type=str, help='path to star/txt/mrcs file.')
    @click.option('-r','--rank',type=int, help='rank of covariance to be estimated.')
    @click.option('-o','--output-dir',type=str,help='path to output directory. when not provided a `result_data` directory will be used with the same path as the provided starfile')
    @click.option('-p','--poses',type=str,default=None,help='Path to pkl file containing particle pose information in cryoDRGN format')
    @click.option('-c','--ctf',type=str,default=None,help='Path to pkl file containing CTF information in cryoDRGN format')
    @click.option('--lazy',is_flag=True,default=False,help='Whether to use lazy dataset. If set, the dataset will not be loaded into (CPU) memory and will be processed when accessed. This is useful for large datasets.')
    @click.option('-w','--whiten',type=bool,default=True,help='whether to whiten the images before processing')
    @click.option('--mask',type=str,default='fuzzy',help="Type of mask to be used on the dataset. Can be either 'fuzzy' or path to a volume file/ Defaults to 'fuzzy'")
    @click.option('--optimize-pose',is_flag=True,default=False,help = 'Whether to optimize over image pose')
    @click.option('--optimize-contrast',is_flag=True,default=False,help = 'Whether to correct for contrast in particle images (can only be used with --optimize-pose flag)')
    @click.option('--class-vols',type=str,default=None,help='Path to GT volumes directory. Used if provided to log eigen vectors error metrics while training.Additionally, GT embedding is computed and logged')
    @click.option('--gt-pose',type=str,default=None,help='Path to GT pkl pose file (cryoDRGN format). Used if provided to log pose error metrics while training')
    @click.option('--debug',is_flag = True,default = False, help = 'debugging mode')
    @click.option('--gt-path',type = str,default = None,help = 'Path to pkl file containing GT dataclass')
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
