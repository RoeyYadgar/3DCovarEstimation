from utils import *
from covar_sgd import Covar,CovarDataset
from covar_distributed import trainParallel
from aspire.volume import Volume,LegacyVolume
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.noise import WhiteNoiseEstimator,AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.noise import WhiteNoiseAdder,PinkNoiseAdder
from aspire.source.relion import RelionSource

import scipy
import torch
import pickle
import os
import pandas as pd
from itertools import product

'''
def update_hyperparam_filenames(folder_name,default_params,filename_prefix = ''):
    df = pd.read_csv(os.path.join(folder_name,'results.csv'))
    for default_param_name,defualt_param_val in default_params.items():
        if(default_param_name not in df.columns):
            df[default_param_name] = defualt_param_val

    for i,df_row in df.iterrows():
        newfilename = filename_prefix + "_".join(f"{param_name}{param_vals[param_dict[param_name]]}" for param_name in param_dict if len(params[param_dict[param_name]]) > 1)
        #df.at[i,'filename'] = 
'''

def run_all_hyperparams(init_covar,dataset,folder_name,param_names,params,filename_prefix = ''):
    default_params = {'lr' : 5e-5,'momentum' : 0.9,'optimType' : 'SGD','reg' : 1e-5,'gammaLr' : 1,'gammaReg' : 1, 'batchSize' : 1, 'epochNum' : 10,'orthogonalProjection' : False}
    for default_param_name,default_param_val in default_params.items(): #add default parameters if they aren't in param_names
        if(default_param_name not in param_names):
            param_names.append(default_param_name)
            params.append([default_param_val])

    param_dict = {param_name : index for index,param_name in enumerate(param_names)}
    param_combinations = list(product(*params))
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i,param_vals in enumerate(param_combinations):
        filename = filename_prefix + "_".join(f"{param_name}{param_vals[param_dict[param_name]]}" for param_name in param_dict)
        filepath = os.path.join(folder_name,filename)
        
        if(not os.path.isfile(filepath)):
            try:
                print(f'Running SGD on : {filepath}')
                c = init_covar()
                trainParallel(c,dataset,savepath = filepath,
                                batch_size = param_vals[param_dict['batchSize']],
                                max_epochs = param_vals[param_dict['epochNum']],
                                lr = param_vals[param_dict['lr']],
                                momentum = param_vals[param_dict['momentum']],
                                optim_type = param_vals[param_dict['optimType']],
                                reg = param_vals[param_dict['reg']],
                                gamma_lr = param_vals[param_dict['gammaLr']],
                                gamma_reg = param_vals[param_dict['gammaReg']],
                                orthogonal_projection= param_vals[param_dict['orthogonalProjection']]
                            )
            
                if(filename_prefix == ''):
                    df_row = pd.DataFrame([[filepath] + (list(param_vals))],
                                        columns = ['filename'] + (param_names))   
                else:
                    df_row = pd.DataFrame([[filepath] + (list(param_vals)) + [filename_prefix]],
                                        columns = ['filename'] + (param_names) + ['prefix'])
                appendCSV(df_row,
                            os.path.join(folder_name,'results.csv'))
            except Exception as e:
                print(e)
 
    
def run_classic_alg(filepath,src,rank,basis = None):
    if(not os.path.isfile(filepath)):
        print(f'Running Classical Algo on : {filepath}')

        if(basis == None):
            basis = FBBasis3D((src.L, src.L, src.L))

        noise_estimator = WhiteNoiseEstimator(src, batchSize=500)
        noise_variance = noise_estimator.estimate()
        mean_estimator = MeanEstimator(src,basis)
        mean_est = mean_estimator.estimate()
        covar_estimator = CovarianceEstimator(src, basis, mean_kernel=mean_estimator.kernel)
        covar_est = covar_estimator.estimate(mean_est, noise_variance)
        eigs_est, lambdas_est = aspire.utils.eigs(covar_est, rank)
        covar_dict = {'covar' : covar_est , 'eigenvecs' : eigs_est, 'eigenvals' : lambdas_est}
        with open(filepath,'wb') as file:
           pickle.dump(covar_dict,file)



def rank2_lr_params_test(folder_name = None):

    if(folder_name == None):
        folder_name = 'data/rank2_L15_lr_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

    learning_rate = [1e-2,1e-3,5e-4 ,1e-4 ,5e-5 ,1e-5 ,5e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                    ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

def rank2_gamma_params_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L15_gamma_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

    learning_rate = [1e-4,1e-5]
    momentum = [0.9]
    regularization = [1e-4,1e-5]
    gamma_lr = [1, 0.8, 0.5, 0.2, 0.1]
    gamma_reg = [1, 0.8, 0.5, 0.2, 0.1]

                
    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

def rank2_ctf_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L15_ctf_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

    learning_rate = [1e-2,1e-3,1e-4,1e-5]
    momentum = [0.9]
    regularization = [1e-4,1e-5,1e-3]
    gamma_lr = [1,0.8,0.5]
    gamma_reg = [1]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank2_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L64_test'

    L = 64
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-3,1e-4,5e-5,1e-5,1e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank4_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L64_test'

    L = 64
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-2,1e-3,1e-4,5e-5,1e-5,1e-6,5e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])


def rank4_eigngap_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_eigengap_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    scaling_factor = [[1,1,1,1,1],[10,5,2,1,1]]
    filename_prefix = ['smallEigengap_' , 'largeEigengap_']

    for i in range(len(scaling_factor)):

        voxels *= np.array(scaling_factor[i],dtype = np.float32).reshape((r+1,1,1,1))        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        learning_rate = [1e-5 * 10]
        momentum = [0.9]
        regularization = [1e-5 , 1e-4 ,1e-3]
        gamma_lr = [1]
        gamma_reg = [0.8,0.5]
        orthogonal_projection = [True,False]

        covar_init = lambda : Covar(L,r)
        vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
        dataset = CovarDataset(sim,vectorsGD=vectorsGD)
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','orthogonalProjection'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,orthogonal_projection],filename_prefix[i])

def rank4_lr_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-2,1e-3,5e-4,1e-4,5e-5]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1,0.8,0.5,0.3]
    gamma_reg = [0.8,0.5,0.3]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank4_optim_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_optim_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [10 ** (-i) for i in range(1,10)]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1,0.8,0.5,0.3]
    gamma_reg = [0.8,0.5,0.3]
    optim = ['Adam']

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg','optimType'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,optim])

def rank2_cont_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_L100_test'

    L = 100
    n = 2048
    r = 2
    voxels = Volume.from_vec(scipy.io.loadmat('data/vols.mat')['vols'].transpose())
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-3,1e-4,5e-5,1e-5]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

def rank2_cont_ctf_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_ctf_L100_test'

    L = 100
    n = 2048
    r = 2
    voxels = Volume.from_vec(scipy.io.loadmat('data/vols.mat')['vols'].transpose())
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

   
    learning_rate = [1e-2,1e-3,1e-4,5e-5]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [1 , 0.5 , 0.2]
    gamma_reg = [0.5]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank4_imsize_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_imsize_test'

    L = 15
    imnum = [2048 * (2 ** i) for i in range(4)]
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)
   
    learning_rate = [5e-4]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [0.8]
    gamma_reg = [0.8]
    epochNum = [5]
    
    filename_prefix = ['numIm=' + str(n) + '_' for n in imnum]
    
    for i,n in enumerate(imnum): 
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        covar_init = lambda : Covar(L,r)
        vectorsGD = volsCovarEigenvec(voxels)
        dataset = CovarDataset(sim,vectorsGD=vectorsGD)
        
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','epochNum'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,epochNum],filename_prefix[i])
    
def rank2_cont_ctf_highresolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_ctf_LHigh_test'

    from scipy.spatial.transform import Rotation as spRot
    num_rots = 50
    vol_rots = np.single(spRot.from_euler('z', np.arange(0,num_rots)/num_rots*2*np.pi).as_matrix())
    
    resolutions = [256,512]
    raise Exception('needs rework')
    for i,L in enumerate(resolutions):
        vol_path = f'data/vols_rots_L={L}.bin'
        if(not os.path.isfile(vol_path)):
            vol_highres = Volume.from_vec(scipy.io.loadmat('data/vols512.mat')['vols'].transpose()).downsample(L)
            high_res = vol_highres.resolution
            voxels = Volume(np.zeros((len(vol_rots),high_res,high_res,high_res)),dtype=np.single)
            for j in range(len(vol_rots)):
                print(j)
                voxels[j] = vol_highres.rotate(Rotation.from_matrix(vol_rots[j]))
            voxels -= np.mean(voxels,axis=0)
            vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2)
            pickle.dump({'vols' : voxels, 'eigenvols' : vectorsGD},open(vol_path,'wb'))
            print('saved file')
        else:
            vol_dict = pickle.load(open(vol_path,'rb'))
            voxels = vol_dict['vols']
            vectorsGD = vol_dict['eigenvols']
        n = 2048
        r = 2
      
        
    
        mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])
    
       
        learning_rate = [1e-4]
        momentum = [0.9]
        regularization = [1e-5]
        gamma_lr = [0.5]
        gamma_reg = [0.5]
    
        covar_init = lambda : Covar(L,r)
        vectorsGD = volsCovarEigenvec(voxels)
        dataset = CovarDataset(sim,vectorsGD=vectorsGD)
        run_all_hyperparams(covar_init,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],filename_prefix = f'L={L}_')
        
def rank4_alg_cmp_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_alg_cmp'

    L = 15
    r = 4
    n = 2048
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,)#unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

   
    learning_rate = [5e-4]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [0.8]
    gamma_reg = [0.8]
    
    
    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],'SGD_')
    
    run_classic_alg(os.path.join(folder_name,'classical_covar.bin'),sim,r)

    #Sim with 'balanced' data
    num_reps = int(np.ceil(n/(r+1)))
    rotations = Rotation.generate_random_rotations(num_reps)
    rotations = np.repeat(rotations.angles, r+1,axis=0)[:n]
    states = np.tile(np.array([i+1 for i in range(r+1)]),num_reps)[:n]

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,angles = rotations,states = states)#,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)]
    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels)
    dataset = CovarDataset(sim,vectorsGD=vectorsGD)
    
    run_all_hyperparams(covar_init,dataset,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],'Balanced_SGD_')
    
    run_classic_alg(os.path.join(folder_name,'Balanced_classical_covar.bin'),sim,r)


def rank4_noise_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L128_noise_test'

    L = 128
    n = 32000
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
    var = np.var(sim.images[:].asnumpy())
   
    learning_rate = [1e-3,5e-4,1e-4,1e-5]
    momentum = [0.9]
    regularization = [1e-6]
    gamma_lr = [0.8,0.5]
    gamma_reg = [0.8,0.5]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels)

    dataset = CovarDataset(sim,0,vectorsGD=vectorsGD)
    run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR=inf_')
    snr_vals = [10, 1 , 0.1 , 0.01]
    for snr in snr_vals:
        noise_var = var / snr
        white_noise_adder = WhiteNoiseAdder(noise_var)
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0, noise_adder=white_noise_adder)
        dataset = CovarDataset(sim,noise_var,vectorsGD=vectorsGD)
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR={snr}_')
    
        #sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0, noise_adder=white_noise_adder,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])
        #run_classic_alg(os.path.join(folder_name,f'classical_covar_SNR={snr}.bin'),sim,r)

def KV_dataset_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/KV_dataset/L100'

    L = 100
    n = 16000
    r = 10

    kv_vols = scipy.io.loadmat('data/KV_dataset/KV_data_L100.mat')
    voxels = Volume(kv_vols['vols'])

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
    hetro_power = np.var(sim.images[:].asnumpy() - sim.vol_forward(Volume(kv_vols['mean_vol']),0,n))
   
    learning_rate = [1e-3,1e-5]
    momentum = [0.9]
    regularization = [1e-6]
    gamma_lr = [0.8]
    gamma_reg = [0.8]

    covar_init = lambda : Covar(L,r)
    vectorsGD = np.sqrt(kv_vols['eigenVals']).reshape(r,1,1,1) * kv_vols['eigenVols']
    mean_volume = Volume(kv_vols['mean_vol'])

    dataset = CovarDataset(sim,0,vectorsGD=vectorsGD,mean_volume = mean_volume)
    run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR=inf_')
    snr_vals = [10, 1 , 0.1 , 0.01]
    for snr in snr_vals:
        noise_var = hetro_power / snr
        white_noise_adder = WhiteNoiseAdder(noise_var)
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0, noise_adder=white_noise_adder)
        dataset = CovarDataset(sim,noise_var,vectorsGD=vectorsGD ,mean_volume = mean_volume)
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR={snr}_')

def realistic_data_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank1_L64_realistic'

    L = 64
    r = 1
    '''
    n = 10000
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    mean_voxel = Volume(np.mean(voxels,axis=0))

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(2.1e4, 2.5e4, 7)])
    var = np.var((sim.images[:] - sim.vol_forward(mean_voxel,0,sim.n)).asnumpy())
    '''
    starpath = 'data/rank1_L64_realistic/source.star'
    sim = RelionSource(starpath)
    voxels = Volume.load('data/rank1_L64_realistic/voxels.mrc')
    mean_voxel = Volume.load('data/rank1_L64_realistic/mean_voxel.mrc')

    learning_rate = [5e-4,1e-4,1e-5,1e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [0.8]
    gamma_reg = [0.8]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels-mean_voxel)

    aiso_noise_estimator = AnisotropicNoiseEstimator(sim)
    sim = sim.whiten(aiso_noise_estimator)

    noise_estimator = WhiteNoiseEstimator(sim, batchSize=500)
    noise_variance = noise_estimator.estimate()

    dataset = CovarDataset(sim,noise_variance,vectorsGD=vectorsGD,mean_volume=mean_voxel)
    run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','batchSize'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,[32]])
    '''
    dataset = CovarDataset(sim,0,vectorsGD=vectorsGD,mean_volume=mean_voxel)
    run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR=inf_')
    
    snr_vals = [0.1 , 0.01]
    for snr in snr_vals:
        noise_var = var / snr
        #noise_adder = PinkNoiseAdder(noise_var)
        noise_adder = WhiteNoiseAdder(noise_var)
        sim = Simulation(n = n , vols = voxels,amplitudes= 1, noise_adder=noise_adder,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(2.1e4, 2.5e4, 7)])
        aiso_noise_estimator = AnisotropicNoiseEstimator(sim)
        sim = sim.whiten(aiso_noise_estimator)
        dataset = CovarDataset(sim,noise_var,vectorsGD=vectorsGD,mean_volume=mean_voxel)
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],f'SNR={snr}_')
    '''

def realistic_data2_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank1_L64_realistic2'

    L = 64
    r = 1
    
    n = 10000
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    mean_voxel = Volume(np.mean(voxels,axis=0))

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(2.1e4, 2.5e4, 7)])
    var = np.var((sim.images[:] - sim.vol_forward(mean_voxel,0,sim.n)).asnumpy())

    learning_rate = [5e-4,1e-4,1e-5,1e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [0.8]
    gamma_reg = [0.8]
    batchSize = [32]

    covar_init = lambda : Covar(L,r)
    vectorsGD = volsCovarEigenvec(voxels-mean_voxel)


    dataset = CovarDataset(sim,0,vectorsGD=vectorsGD,mean_volume=mean_voxel)
    run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','batchSize'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,batchSize],f'SNR=inf_')
    
    snr_vals = [0.1 , 0.01]
    for snr in snr_vals:
        noise_var = var / snr
        #noise_adder = PinkNoiseAdder(noise_var)
        noise_adder = WhiteNoiseAdder(noise_var)
        sim = Simulation(n = n , vols = voxels,amplitudes= 1, noise_adder=noise_adder,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(2.1e4, 2.5e4, 7)])
        aiso_noise_estimator = AnisotropicNoiseEstimator(sim)
        sim = sim.whiten(aiso_noise_estimator)
        dataset = CovarDataset(sim,noise_var,vectorsGD=vectorsGD,mean_volume=mean_voxel)
        run_all_hyperparams(covar_init,dataset,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','batchSize'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,batchSize],f'SNR={snr}_')


if __name__ == "__main__":
    '''
    rank2_lr_params_test()
    rank2_gamma_params_test()
    rank2_ctf_test()
    rank2_resolution_test()
    rank4_resolution_test()
    rank4_eigngap_test()
    rank4_lr_test()
    rank4_optim_test()
    rank2_cont_resolution_test()
    #rank2_cont_ctf_resolution_test()
    rank4_imsize_test()
    '''
    #rank4_noise_test()
    #KV_dataset_test()
    realistic_data_test()
    realistic_data2_test()
    
