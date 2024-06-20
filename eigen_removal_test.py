#%%
import torch
import pickle
import numpy as np
from covar_sgd import CovarDataset, Covar,trainCovar
from covar_distributed import trainParallel
from aspire.volume import Volume,LegacyVolume
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from aspire.image import Image
from wiener_coords import wiener_coords
from utils import estimateMean,volsCovarEigenvec
L = 64
r = 2
vols = LegacyVolume(
            L=L,
            C=3,
            dtype=np.float32,
        ).generate()

source = Simulation(
    n=2048,
    vols=vols,
    dtype=np.float32,
    amplitudes=1,
    offsets = 0,
    unique_filters=[RadialCTFFilter(defocus=d,pixel_size = 2.8) for d in np.linspace(1.5e4, 2.5e4, 7)]
    )

#mean_est = estimateMean(source)
mean_est = Volume(np.mean(vols,axis=0))
covar_eigenvecs = volsCovarEigenvec(vols)

dataset = CovarDataset(source,0,vectorsGD=covar_eigenvecs,mean_volume=mean_est)
device = torch.device('cuda:0')

#%% Test Iterative eigen training
from iterative_covar_sgd import IterativeCovar,IterativeCovarTrainer
cov = IterativeCovar(L,r)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 8)
device = torch.device('cuda:0')
covar_model = cov.to(device)
trainer = IterativeCovarTrainer(covar_model,dataloader,device)
trainer.train(5,lr = 1e-3,optim_type = 'Adam',reg = 1e-5,gamma_lr = 0.8,gamma_reg = 0.8,orthogonal_projection = True)

#%%

eigenvals_GD = np.linalg.norm(covar_eigenvecs,axis=1) ** 2
eigenvectors_GD = (covar_eigenvecs / np.sqrt(eigenvals_GD[:,np.newaxis])).reshape((-1,L,L,L))
coords_GD = wiener_coords(dataset,torch.tensor(eigenvectors_GD).to('cuda:0'),torch.tensor(eigenvals_GD).to('cuda:0'),8)

#new_dataset = dataset.remove_vol_from_images(torch.tensor(eigenvectors_GD).to('cuda:0'),coords_GD,copy_dataset=True)
#%%

'''
with open('data_dump.pkl','rb') as fid:
    data_dict = pickle.load(fid)


dataset = data_dict['dataset']
eigen_est = data_dict['eigen_est']
eigenval_est = data_dict['eigenval_est']
coords = data_dict['coords']
covar_eigenvecs = data_dict['covar_eigenvecs']

'''


from covar_sgd import trainCovar,Covar
from covar_distributed import trainParallel
cov = Covar(L,2)
trainCovar(cov,dataset,savepath = None,
                batch_size = 32,
                max_epochs = 5,
                lr = 1e-2/10,optim_type = 'Adam',
                reg = 1e-6*10,
                gamma_lr = 0.8,
                gamma_reg = 0.8,
                orthogonal_projection= True)


# %% Remove first eigenvector from dataset and train Again
new_dataset = dataset.remove_vol_from_images(torch.tensor(eigenvectors_GD).to('cuda:0')[:1],coords_GD[:,:1],copy_dataset=True)

cov_new = Covar(L,1)
trainCovar(cov_new,new_dataset,savepath = None,
                batch_size = 32,
                max_epochs = 5,
                lr = 1e-2/10,optim_type = 'Adam',
                reg = 1e-6*10,
                gamma_lr = 0.8,
                gamma_reg = 0.8,
                orthogonal_projection= True)
# %% 
dataset = CovarDataset(source,0,vectorsGD=covar_eigenvecs,mean_volume=mean_est)
cov = Covar(L,1)
trainCovar(cov,dataset,savepath = None,
                batch_size = 32,
                max_epochs = 5,
                lr = 1e-2/10,optim_type = 'Adam',
                reg = 1e-6*10,
                gamma_lr = 0.8,
                gamma_reg = 0.8,
                orthogonal_projection= True)
eigenvectors,eigenvals = cov.eigenvecs
coords = wiener_coords(dataset,eigenvectors,eigenvals)
dataset = dataset.remove_vol_from_images(eigenvectors[:1],coords[:,:1])
cov = Covar(L,1)
trainCovar(cov,dataset,savepath = None,
                batch_size = 32,
                max_epochs = 5,
                lr = 1e-2/10,optim_type = 'Adam',
                reg = 1e-6*10,
                gamma_lr = 0.8,
                gamma_reg = 0.8,
                orthogonal_projection= True)
new_eigenvectors,new_eigenvals = cov.eigenvecs

# %%
