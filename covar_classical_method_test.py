import logging

import numpy as np
from scipy.cluster.vq import kmeans2

from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.denoising import src_wiener_coords
from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils import eigs
from aspire.utils.random import Random
from aspire.volume import LegacyVolume, Volume
import pickle
from utils import volsCovarEigenvec

logger = logging.getLogger(__name__)

# Specify parameters
img_size = 15  # image size in square
num_imgs = 2048  # number of images
num_eigs = 4  # number of eigen-vectors to keep
dtype = np.float32

# Generate a ``Volume`` object for use in the simulation. Here we use a ``LegacyVolume`` and
# set C = 3 to generate 3 unique random volumes.
vols = LegacyVolume(
    L=img_size,
    C=5,
    dtype=dtype,
).generate()

# Create a simulation object with specified filters
sim = Simulation(
    unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
    n=num_imgs,
    vols=vols,
    dtype=dtype,
    amplitudes=1,
    offsets = 0
)
# The Simulation object was created using 3 volumes.
num_vols = sim.C

# Specify the normal FB basis method for expending the 2D images
basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_estimator = WhiteNoiseEstimator(sim, batchSize=500)
noise_variance = noise_estimator.estimate()
logger.info(f"Noise Variance = {noise_variance}")


mean_estimator = MeanEstimator(sim, basis)
mean_est = mean_estimator.estimate()

# Passing in a mean_kernel argument to the following constructor speeds up some calculations
covar_estimator = CovarianceEstimator(sim, basis, mean_kernel=mean_estimator.kernel)
covar_est = covar_estimator.estimate(mean_est, noise_variance)



filename = 'data/classical_covar_est_rank4.bin'
with open(filename,'wb') as file:
    pickle.dump(covar_est,file)
    
eigs_est, lambdas_est = eigs(covar_est, num_eigs)

eigs_est = eigs_est.transpose((3,0,1,2)).reshape((num_eigs,-1))   
eigs_gd = volsCovarEigenvec(vols.asnumpy()).asnumpy().reshape((num_eigs,-1))
sing_vals = np.linalg.norm(eigs_gd,axis=1).reshape((-1,1))
eigs_gd = eigs_gd / sing_vals

cosine_sim = np.matmul(eigs_est,eigs_gd.transpose())

cosine_sim_mean = np.mean(np.sqrt(np.sum(cosine_sim ** 2,axis = 1)))
cosine_sim_weighted_mean = np.sum(np.sqrt(np.sum(cosine_sim ** 2,axis = 1)) * sing_vals.transpose())/np.sum(sing_vals)

print(f'Cosine Sim Mean of classical method : {cosine_sim_mean}. Cosine Sim Weighted Mean with sqrt(eigen_vals) : {cosine_sim_weighted_mean} ')