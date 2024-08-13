import unittest
from aspire.volume import Volume,LegacyVolume
from aspire.source import Simulation
from covar_sgd import CovarDataset,Covar
from aspire.operators import RadialCTFFilter
from aspire.denoising import src_wiener_coords
import time
from wiener_coords import wiener_coords,latentMAP
import torch
import numpy as np

class TestWienerImpl(unittest.TestCase):

    def test_wiener_impl(self):
        L = 64
        r = 2

        vols = LegacyVolume(
                    L=L,
                    C=3,
                    dtype=np.float32,
                ).generate()

        source = Simulation(
            n=100,
            vols=vols,
            dtype=np.float32,
            amplitudes=1,
            offsets = 0,
            unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)]
            )
        mean_vol = Volume(np.mean(vols,axis=0))
        noise_var = 1
        dataset = CovarDataset(source,noise_var,vectorsGD=None,mean_volume=mean_vol)
        device = torch.device('cuda:0')


        cov = Covar(L,r).to(device)
        eigenvecs,eigenvals = cov.eigenvecs

        t = time.time()
        coords = wiener_coords(dataset,eigenvecs,eigenvals)
        elapsed_time_impl = time.time() - t

        t = time.time()
        coords2 = latentMAP(dataset,eigenvecs,eigenvals)
        elapsed_time_implMap = time.time() - t

        eigenvecs = Volume(eigenvecs.cpu().numpy())
        eigenvals = np.diag(eigenvals.cpu().numpy())
        t = time.time()
        coords_aspire = src_wiener_coords(source,mean_vol,eigenvecs,eigenvals,noise_var = noise_var)
        elapsed_time_aspire = time.time() - t

        np.testing.assert_allclose(coords.cpu().numpy(),coords_aspire.T,rtol = 1e-6,atol = 1e-2)
        torch.testing.assert_close(coords,coords2,rtol=1e-6,atol=1e-2)
        print(f'Elapsed time of aspire implementation : {elapsed_time_aspire}')
        print(f'Elapsed time of own implementation : {elapsed_time_impl}')
        print(f'Elapsed time of MAP implementation : {elapsed_time_implMap}')



if __name__ == "__main__":
    unittest.main()




