import gzip
import os
import shutil
import unittest

import numpy as np
import requests
import torch
from aspire.utils import grid_3d
from aspire.volume import LegacyVolume, Volume
from matplotlib import pyplot as plt

from cov3d.dataset import CovarDataset
from cov3d.mean import reconstruct_mean, reconstruct_mean_from_halfsets
from cov3d.projection_funcs import centered_fft3, centered_ifft3
from cov3d.source import SimulatedSource


def download_mrc(emd_id: int, output_path: str):
    # Build URL
    base_ftp = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures"
    url = f"{base_ftp}/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
    output_path = os.path.abspath(output_path)
    output_gz = f"{output_path}.gz"
    output_dir = os.path.dirname(output_path)

    # Make directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading from {url} ...")

    # Stream download so large files don't overwhelm memory
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_gz, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

    with gzip.open(output_gz, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(output_gz)

    print(f"Saved to {output_path}")
    return output_path


def display_projections(volume):
    volume = volume.squeeze()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    volume = volume.squeeze()
    axes = ["X", "Y", "Z"]
    for i, ax in enumerate(axs):
        if i == 0:
            proj = volume.sum(dim=0).cpu().numpy()
        elif i == 1:
            proj = volume.sum(dim=1).cpu().numpy()
        else:
            proj = volume.sum(dim=2).cpu().numpy()
        im = ax.imshow(proj, cmap="viridis")
        ax.set_title(f"Projection along {axes[i]}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return fig


class TestYourClassName(unittest.TestCase):

    def _gen_rand_volume(self):
        voxels = LegacyVolume(L=self.L, C=1, K=64, dtype=self.dtype_np).generate()
        # Create a smooth radially symmetric volume: 1 at center, 0 for radius >= L/2
        L = self.L
        # Make 3D coords (centered)
        grid = np.arange(L) - (L - 1) / 2
        zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        # Scale radius to [0,1] (r=0 center, r=0.5*L edge), use smooth cutoff (e.g. cosine taper)
        rnorm = (rr / (L / 2)) * 3
        smooth = np.zeros_like(rnorm)
        inside = rnorm <= 1
        # Use raised cosine (smoothly goes from 1 to 0)
        smooth[inside] = 0.5 * (1 + np.cos(np.pi * rnorm[inside]))
        voxels = voxels * smooth[np.newaxis, ...].astype(self.dtype_np)
        return voxels

    def _gen_volume(self):
        vol = Volume.load("/home/ry295/pi_data/igg_1d/vols/128_org/000.mrc", dtype=self.dtype_np)
        if vol.shape[-1] > self.L:
            vol = vol.downsample(self.L)

        filter = self._get_gaussian_filter(0.2)

        filter *= torch.tensor(grid_3d(self.L, shifted=False, normalized=True)["r"] <= 1)

        vol_tensor = centered_fft3(torch.tensor(vol.asnumpy()))
        vol_tensor = centered_ifft3(filter * vol_tensor).real

        vol = Volume(vol_tensor.numpy())

        return vol

    def _get_gaussian_filter(self, sigma=0.5):
        """Returns a 3D Gaussian filter matching the dataset volume size."""
        L = self.L

        grid = grid_3d(L, shifted=False, normalized=False)
        xx, yy, zz = (grid["x"], grid["y"], grid["z"])

        gaussian = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * (L * sigma) ** 2))
        return torch.tensor(gaussian, dtype=self.dtype)

    def _gen_dataset(self):
        vol = self._gen_volume()
        src = SimulatedSource(n=self.n, vols=vol, noise_var=0, whiten=False)
        return CovarDataset(src, noise_var=0, apply_preprocessing=False), vol

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.L = 128
        self.n = 30000
        self.upsampling_factor = 2
        self.dtype = torch.float64
        self.dtype_np = np.float64
        dataset, source_vol = self._gen_dataset()
        self.dataset = dataset
        self.source_vol = torch.tensor(source_vol.asnumpy())

    def tearDown(self):
        # Clean up after tests
        pass

    def test_reconstruct_mean_clean_dataset(self):
        reconstructed_mean, rhs, lhs = reconstruct_mean(
            self.dataset, upsampling_factor=self.upsampling_factor, return_lhs_rhs=True
        )
        reconstructed_mean = reconstructed_mean.to("cpu")
        rhs = rhs.to("cpu")
        lhs = lhs.to("cpu")

        source_vol_fourier = centered_fft3(self.source_vol).squeeze()
        reconstructed_mean_fourier = centered_fft3(reconstructed_mean)

        if self.L % 2 == 0:
            # For even size we do not recover the negative fourier elements that don't have a positive counterpart
            # i.e. -N/2, -N/2 + 1, ..., N/2-1 we disregard -N/2 from the mask
            rhs = rhs[1:, 1:, 1:]
            lhs = lhs[1:, 1:, 1:]

            source_vol_fourier = source_vol_fourier[1:, 1:, 1:]
            reconstructed_mean_fourier = reconstructed_mean_fourier[1:, 1:, 1:]

        torch.testing.assert_close(
            reconstructed_mean_fourier, source_vol_fourier, rtol=1e-8, atol=1e-4 * source_vol_fourier.abs().max()
        )

        regularized_reconstructed_mean = reconstruct_mean_from_halfsets(
            self.dataset, upsampling_factor=self.upsampling_factor
        )
        regularized_reconstructed_mean = regularized_reconstructed_mean.to("cpu")
        relative_error = torch.norm(reconstructed_mean - self.source_vol) / torch.norm(self.source_vol)
        relative_error_regularized = torch.norm(regularized_reconstructed_mean - self.source_vol) / torch.norm(
            self.source_vol
        )
        print(f"Relative error {relative_error} (unregularized) {relative_error_regularized} (regularized)")


if __name__ == "__main__":
    unittest.main()
