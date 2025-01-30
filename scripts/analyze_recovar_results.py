import pickle
import numpy as np
import torch
import click
import os
from aspire.volume import Volume
from cov3d.covar_sgd import frobeniusNormDiff,frobeniusNorm,cosineSimilarity

@click.command()
@click.option('--recovar-results', type=click.Path(exists=True))
@click.option('--cov3d-results', type=click.Path(exists=True))
def compare_recovar_results(recovar_results,cov3d_results):
    cov3d_results = torch.load(os.path.join(cov3d_results,'training_results.bin'))
    cov3d_eigenvectors = cov3d_results['_vectors_real']
    gd_eigenvectors = cov3d_results['vectorsGD']
    device = cov3d_eigenvectors.device
    rank = cov3d_eigenvectors.shape[0]
    L = cov3d_eigenvectors.shape[1]

    with open(os.path.join(recovar_results,'model/eigenvals.pkl'), 'rb') as f:
        recovar_eigenvalues = pickle.load(f)
        #recovar_eigenvalues = recovar_results['s']
    
    recovar_eigenvectors = Volume(np.zeros((rank,L,L,L)))
    for i in range(rank):
        volume_path = f'output/volumes/eigen_pos{i:04d}.mrc'
        recovar_eigenvectors[i] = Volume.load(os.path.join(recovar_results,volume_path)) * np.sqrt(recovar_eigenvalues[i])

    recovar_eigenvectors = torch.tensor(recovar_eigenvectors.asnumpy(),device=device,dtype=cov3d_eigenvectors.dtype)
    recovar_eigenvectors = recovar_eigenvectors.reshape(rank,-1)
    cov3d_eigenvectors = cov3d_eigenvectors.reshape(rank,-1)
    frobenius_norm_err_recovar = frobeniusNormDiff(recovar_eigenvectors*L, gd_eigenvectors)/frobeniusNorm(gd_eigenvectors)
    frobenius_norm_err_cov3d = frobeniusNormDiff(cov3d_eigenvectors, gd_eigenvectors)/frobeniusNorm(gd_eigenvectors)

    cosine_sim_recovar = cosineSimilarity(recovar_eigenvectors,gd_eigenvectors)
    cosine_sim_cov3d = cosineSimilarity(cov3d_eigenvectors,gd_eigenvectors)

    cosine_sim_recovar = np.mean(np.sqrt(np.sum(cosine_sim_recovar ** 2,axis = 0)))
    cosine_sim_cov3d = np.mean(np.sqrt(np.sum(cosine_sim_cov3d ** 2,axis = 0)))


    print(f'Frobenius norm error for Recovar: {frobenius_norm_err_recovar}')
    print(f'Frobenius norm error for Cov3D: {frobenius_norm_err_cov3d}')
    print(f'Cosine similarity for Recovar: {cosine_sim_recovar}')
    print(f'Cosine similarity for Cov3D: {cosine_sim_cov3d}')


if __name__ == "__main__":
    compare_recovar_results()