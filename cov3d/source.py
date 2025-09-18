import os
import pickle
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from cryodrgn.ctf import compute_ctf, load_ctf_for_training
from cryodrgn.source import ImageSource as CryoDRGNImageSource

from cov3d.poses import get_phase_shift_grid, pose_cryoDRGN2APIRE
from cov3d.projection_funcs import centered_fft2, centered_ifft2


class ImageSource:
    """Image source for cryo-EM particle images with CTF and pose information.

    Provides access to particle images along with their corresponding CTF parameters,
    rotations, and translations. Supports preprocessing including whitening and normalization.
    Uses cryoDRGN image source under the hood to read particle images and associated data.

    Attributes:
        particles_path: Path to particle images file
        ctf_path: Path to CTF parameters file
        poses_path: Path to poses file
        device: Device for computations
        image_source: CryoDRGN image source
        dtype: Data type for tensors
        ctf_params: CTF parameters for each particle
        freq_lattice: Frequency lattice for CTF computation
        indices: Indices of particles to use
        rotations: Rotation matrices for each particle
        offsets: Translation offsets for each particle
        apply_preprocessing: Whether to apply preprocessing
        whitening_filter: Whitening filter for noise reduction
        offset_normalization: Per-image offset normalization
        scale_normalization: Per-image scale normalization
    """

    def __init__(
        self,
        particles_path: str,
        ctf_path: Optional[str] = None,
        poses_path: Optional[str] = None,
        indices: Optional[torch.Tensor] = None,
        apply_preprocessing: bool = True,
    ) -> None:
        """Initialize ImageSource.

        Args:
            particles_path: Path to particle images file
            ctf_path: Path to CTF parameters file (optional)
            poses_path: Path to poses file (optional)
            indices: Indices of particles to use (optional)
            apply_preprocessing: Whether to apply preprocessing (default: True)
        """
        self.particles_path = particles_path
        self.device = torch.device("cpu")
        self.image_source = CryoDRGNImageSource.from_file(self.particles_path, indices=indices)
        if self.image_source.dtype == "float32":
            self.dtype = torch.float32
        elif self.image_source.dtype == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self.image_source.dtype}. Only float32 and float64 are supported.")

        # If ctf or poses were not provided check if they exist in the same dir as the particles file
        particles_dir = os.path.split(self.particles_path)[0]
        if ctf_path is None:
            ctf_path = os.path.join(particles_dir, "ctf.pkl")
            assert os.path.isfile(
                ctf_path
            ), f"ctf file was not provided, tried {ctf_path} as a default but file does not exist"
        if poses_path is None:
            poses_path = os.path.join(particles_dir, "poses.pkl")
            assert os.path.isfile(
                poses_path
            ), f"poses file was not provided, tried {poses_path} as a default but file does not exist"
        self.ctf_path = ctf_path
        self.poses_path = poses_path

        self.ctf_params = torch.tensor(load_ctf_for_training(self.resolution, ctf_path))
        self.freq_lattice = (
            (torch.stack(get_phase_shift_grid(self.resolution), dim=0) / torch.pi / 2)
            .permute(2, 1, 0)
            .reshape(self.resolution**2, 2)
        )

        if indices is None:
            indices = torch.arange(self.image_source.n)
        self.indices = indices
        self.ctf_params = self.ctf_params[indices]

        with open(poses_path, "rb") as f:
            poses = pickle.load(f)
        rots, offsets = pose_cryoDRGN2APIRE(poses, self.resolution)
        self.rotations = torch.tensor(rots.astype(self.image_source.dtype))[indices]
        self.offsets = torch.tensor(offsets.astype(self.image_source.dtype))[indices]
        self.apply_preprocessing = apply_preprocessing

        self.whitening_filter = None
        self.offset_normalization = torch.zeros(self.image_source.n)
        self.scale_normalization = torch.ones(self.image_source.n)
        if self.apply_preprocessing:
            self._preprocess_images()

    @property
    def resolution(self) -> int:
        """Get image resolution.

        Returns:
            Image resolution
        """
        return self.image_source.D

    def __len__(self) -> int:
        """Get number of particles.

        Returns:
            Number of particles
        """
        return self.image_source.n

    def to(self, device: Union[torch.device, str, None]) -> "ImageSource":
        """Move the ImageSource to the specified device (CPU/GPU).

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        if device is None:
            device = torch.device("cpu")

        self.device = device

        # Move tensors to device
        self.ctf_params = self.ctf_params.to(device)
        self.freq_lattice = self.freq_lattice.to(device)
        self.offset_normalization = self.offset_normalization.to(device)
        self.scale_normalization = self.scale_normalization.to(device)

        # Move whitening filter if it exists
        if self.whitening_filter is not None:
            self.whitening_filter = self.whitening_filter.to(device)

        return self

    def get_ctf(self, index: Union[int, torch.Tensor]) -> torch.Tensor:
        """Get CTF for given particle indices.

        Args:
            index: Particle index or indices

        Returns:
            CTF values
        """
        ctf_params = self.ctf_params[index]
        freq_lattice = self.freq_lattice / ctf_params[:, 0].view(-1, 1, 1)
        ctf = compute_ctf(freq_lattice, *torch.split(ctf_params[:, 1:], 1, 1)).reshape(
            -1, self.resolution, self.resolution
        )

        return ctf if not self.apply_preprocessing else ctf * self.whitening_filter

    def images(self, index: Union[int, torch.Tensor], fourier: bool = False) -> torch.Tensor:
        """Get particle images for given indices.

        Args:
            index: Particle index or indices
            fourier: Whether to return in Fourier domain (default: False)

        Returns:
            Particle images
        """
        images = self.image_source.images(index)
        images = images.to(self.device)
        if not self.apply_preprocessing and not fourier:
            return images

        images = centered_fft2(images)

        if self.apply_preprocessing:
            images *= self.whitening_filter
            images[:, self.resolution // 2, self.resolution // 2] -= (
                self.offset_normalization[index] * self.resolution**2
            )
            images /= self.scale_normalization[index].reshape(-1, 1, 1)

        if not fourier:
            images = centered_ifft2(images).real

        return images

    def __getitem__(
        self, index: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get particle data for given index.

        Args:
            index: Particle index

        Returns:
            Tuple of (images, ctf, rotations, offsets)
        """
        return self.images(index), self.get_ctf(index), self.rotations[index], self.offsets[index]

    def _preprocess_images(self, batch_size: int = 1024) -> None:
        """Preprocess images with whitening and normalization.

        Whitens images by estimating the noise PSD and apply it as a filter on all images.
        Additionally each image is normalized individually to have N(0,1) background noise.
        Implementation is based on ASPIRE:
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/noise/noise.py#L333
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/image/image.py#L27

        Args:
            batch_size: Batch size for processing (default: 1024)
        """
        mask = (torch.norm(self.freq_lattice, dim=1) >= 0.5).reshape(self.resolution, self.resolution)
        n = len(self)
        mean_est = 0
        noise_psd_est = torch.zeros((self.resolution,) * 2)
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            # Use original unaltered images (not self.images)
            images = self.image_source.images(idx) * mask

            mean_est += torch.sum(images)
            noise_psd_est += torch.sum(torch.abs(centered_fft2(images)) ** 2, dim=0)

        mean_est /= torch.sum(mask) * n
        noise_psd_est /= torch.sum(mask) * n

        noise_psd_est[self.resolution // 2, self.resolution // 2] -= mean_est**2

        self.whitening_filter = (1 / torch.sqrt(noise_psd_est)).unsqueeze(0)

        # Per-image normalization
        # After setting up whitening filter, we can access self.images to get the whitened images
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            images = self.images(idx)
            mean = torch.mean(images[:, mask], dim=1)
            std = torch.std(images[:, mask], dim=1)
            self.offset_normalization[idx] = mean
            self.scale_normalization[idx] = std

    def estimate_noise_var(self, batch_size: int = 1024) -> float:
        """Estimate noise variance from images.

        Args:
            batch_size: Batch size for processing (default: 1024)

        Returns:
            Estimated noise variance
        """
        mask = (torch.norm(self.freq_lattice, dim=1) >= 0.5).reshape(self.resolution, self.resolution)
        n = len(self)
        first_moment = 0
        second_moment = 0
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            images = self.images(idx)
            images_masked = images * mask

            first_moment += torch.sum(images_masked)
            second_moment += torch.sum(torch.abs(images_masked) ** 2)

        first_moment /= torch.sum(mask) * n
        second_moment /= torch.sum(mask) * n
        return second_moment - first_moment**2

    def get_subset(self, idx: Union[int, torch.Tensor]) -> "ImageSource":
        """Get subset of particles.

        Args:
            idx: Indices of particles to include

        Returns:
            New ImageSource with subset of particles
        """
        subset = deepcopy(self)
        subset.indices = subset.indices[idx]
        subset.image_source = CryoDRGNImageSource.from_file(subset.particles_path, indices=subset.indices)
        subset.ctf_params = subset.ctf_params[idx]
        subset.rotations = subset.rotations[idx]
        subset.offsets = subset.offsets[idx]

        subset.scale_normalization = subset.scale_normalization[idx]
        subset.offset_normalization = subset.offset_normalization[idx]

        return subset

    def get_paths(self) -> Tuple[str, str, str]:
        """Get file paths.

        Returns:
            Tuple of (particles_path, ctf_path, poses_path)
        """
        return self.particles_path, self.ctf_path, self.poses_path
