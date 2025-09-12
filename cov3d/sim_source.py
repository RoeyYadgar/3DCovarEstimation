import logging
import os
import pickle

import mrcfile
import numpy as np
import torch
from aspire.image import Image
from aspire.operators import ArrayFilter, MultiplicativeFilter, RadialCTFFilter, ScalarFilter
from aspire.utils import Rotation
from aspire.volume import LegacyVolume, Volume, rotated_grids
from matplotlib import pyplot as plt

from cov3d.analyze import analyze
from cov3d.dataset import CovarDataset, GTData
from cov3d.nufft_plan import NufftPlan
from cov3d.poses import pose_ASPIRE2cryoDRGN
from cov3d.projection_funcs import vol_forward
from cov3d.utils import get_torch_device, readVols, volsCovarEigenvec
from cov3d.workflow import covar_processing, load_mask

logger = logging.getLogger(__name__)


class SimulatedSource:
    def __init__(self, n, vols, noise_var, whiten=True, unique_filters=None, rotations_std=0, offsets_std=0):
        self.n = n
        self.L = vols.shape[-1]
        self.num_vols = vols.shape[0]
        self.vols = vols
        self.whiten = whiten
        if unique_filters is None:
            unique_filters = [ArrayFilter(np.ones((self.L, self.L)))]
        self._unique_filters = unique_filters
        self.rotations_std = rotations_std
        self.offsets_std = offsets_std
        self._clean_images = self._gen_clean_images()
        self.noise_var = noise_var

    @property
    def noise_var(self):
        return self._noise_var if (not self.whiten) else 1

    @noise_var.setter
    def noise_var(self, noise_var):
        self._noise_var = noise_var
        self._image_noise = torch.randn(
            (self.n, self.L, self.L), dtype=self._clean_images.dtype, device=self._clean_images.device
        ) * (self._noise_var**0.5)

    @property
    def images(self):
        images = self._clean_images + self._image_noise
        if self.whiten:
            images /= (self._noise_var) ** 0.5

        return Image(images.numpy())

    @property
    def unique_filters(self):
        whiten_filter = ScalarFilter(dim=2, value=self._noise_var ** (-0.5))
        return [MultiplicativeFilter(filt, whiten_filter) for filt in self._unique_filters]

    def noisify_rotations(self, rots, noise_std):
        noisy_rots = Rotation.from_matrix(rots).as_rotvec()
        noisy_rots += noise_std * np.random.randn(*noisy_rots.shape)
        return Rotation.from_rotvec(noisy_rots).matrices.astype(rots.dtype)

    def _gen_clean_images(self, batch_size=1024):
        clean_images = torch.zeros((self.n, self.L, self.L))
        self._offsets = torch.zeros((self.n, 2))  # TODO: create non-zero gt offsets
        self.offsets = self._offsets + self.L * self.offsets_std * np.random.randn(self.n, 2)
        self.amplitudes = np.ones((self.n))
        self.states = torch.tensor(np.random.choice(self.num_vols, self.n))
        self.filter_indices = np.random.choice(len(self._unique_filters), self.n)
        self._rotations = Rotation.generate_random_rotations(self.n).matrices
        self.rotations = self.noisify_rotations(self._rotations, self.rotations_std)

        unique_filters = torch.tensor(
            np.array([self._unique_filters[i].evaluate_grid(self.L) for i in range(len(self._unique_filters))]),
            dtype=torch.float32,
        )
        pts_rot = torch.tensor(rotated_grids(self.L, self._rotations).copy()).reshape((3, self.n, self.L**2))
        pts_rot = pts_rot.transpose(0, 1)
        pts_rot = torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi

        device = get_torch_device()
        volumes = torch.tensor(self.vols.asnumpy(), device=device)
        nufft_plan = NufftPlan((self.L,) * 3, batch_size=1, dtype=volumes.dtype, device=device)

        for i in range(self.num_vols):
            idx = (self.states == i).nonzero().reshape(-1)
            for j in range(0, len(idx), batch_size):
                batch_ind = idx[j : j + batch_size]
                ptsrot = pts_rot[batch_ind].to(device)
                filter_indices = self.filter_indices[batch_ind]
                filters = unique_filters[filter_indices].to(device)

                nufft_plan.setpts(ptsrot)
                projected_volume = vol_forward(volumes[i].unsqueeze(0), nufft_plan, filters).squeeze(1)

                clean_images[batch_ind] = projected_volume.cpu()

        return clean_images

    def _ctf_cryodrgn_format(self):
        ctf = np.zeros((len(self._unique_filters), 9))
        for i, ctf_filter in enumerate(self._unique_filters):
            ctf[i, 0] = self.L
            ctf[i, 1] = ctf_filter.pixel_size
            ctf[i, 2] = ctf_filter.defocus_u
            ctf[i, 3] = ctf_filter.defocus_v
            ctf[i, 4] = ctf_filter.defocus_ang / np.pi * 180
            ctf[i, 5] = ctf_filter.voltage
            ctf[i, 6] = ctf_filter.Cs
            ctf[i, 7] = ctf_filter.alpha
            ctf[i, 8] = 0  # phase shift

        full_ctf = np.zeros((self.n, 9))
        for i in range(ctf.shape[0]):
            full_ctf[self.filter_indices == i] = ctf[i]

        return full_ctf

    def save(self, output_dir, file_prefix=None, save_image_stack=True, gt_pose=False, whiten=False):

        def add_prefix(filename):
            return f"{file_prefix}_{filename}" if file_prefix is not None else filename

        mrcs_output = os.path.join(output_dir, add_prefix("particles.mrcs"))
        poses_output = os.path.join(output_dir, add_prefix("poses.pkl"))
        ctf_output = os.path.join(output_dir, add_prefix("ctf.pkl"))

        if save_image_stack:
            whiten_val = self.whiten
            self.whiten = whiten
            with mrcfile.new(mrcs_output, overwrite=True) as mrc:
                mrc.set_data(self.images.asnumpy().astype(np.float32))
                # mrc.voxel_size = self.vols.pixel_size
                # mrc.set_spacegroup(1)
                # mrc.data = np.transpose(mrc.data,(0,2,1))
                # mrc.update_header()
            self.whiten = whiten_val

        if gt_pose:
            rots = self._rotations
            offsets = self._offsets
        else:
            rots = self.rotations
            offsets = self.offsets
        poses = pose_ASPIRE2cryoDRGN(rots, offsets, self.L)
        with open(poses_output, "wb") as f:
            pickle.dump(poses, f)

        with open(ctf_output, "wb") as f:
            pickle.dump(self._ctf_cryodrgn_format(), f)


def display_source(source, output_path, num_ims=2, display_clean=False):
    num_vols = len(np.unique(source.states))
    fig, axs = plt.subplots(num_ims, num_vols, figsize=(2 * num_vols, 2 * num_ims))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    im_samples = source._clean_images[:20].numpy() if display_clean else source.images[:20].asnumpy()
    im_min = im_samples.min()
    im_max = im_samples.max()

    for i in range(num_vols):
        state_inds = np.where(source.states == i)[0][:num_ims]
        clean_images = (
            source._clean_images[state_inds].numpy() if display_clean else source.images[state_inds].asnumpy()
        )
        for j in range(num_ims):
            axs_idx = (j, i) if num_vols > 1 else j
            axs[axs_idx].imshow(clean_images[j], cmap="gray", vmin=im_min, vmax=im_max)
            axs[axs_idx].set_xticks([])  # Remove x-axis ticks
            axs[axs_idx].set_yticks([])
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)


def replicate_source(source):
    source.rotations = np.tile(source.rotations, (2, 1, 1))
    source.filter_indices = np.tile(source.filter_indices, (2))
    source.states = torch.tile(source.states, (2,))
    source.amplitudes = np.tile(source.amplitudes, (2))
    source.offsets = torch.tile(source.offsets, (2, 1))
    source._clean_images = torch.tile(source._clean_images, (2, 1, 1))
    source._image_noise = torch.tile(source._image_noise, (2, 1, 1))
    source.n = source.n * 2

    return source


def simulateExp(folder_name=None, L=64, r=5, no_ctf=False, save_source=False, vols=None, mask=None):
    os.makedirs(folder_name, exist_ok=True)

    n = 100000
    pixel_size = 3 * 128 / L

    if not no_ctf:
        filters = [
            RadialCTFFilter(defocus=d, pixel_size=pixel_size)
            for d in np.random.lognormal(np.log(20000), 0.3, size=(928))
        ]
    else:
        filters = [ArrayFilter(np.ones((L, L)))]

    if vols is None:
        voxels = LegacyVolume(L=int(L * 0.7), C=r + 1, K=64, dtype=np.float32, pixel_size=pixel_size).generate()
        padded_voxels = np.zeros((r + 1, L, L, L), dtype=np.float32)
        pad_width = (L - voxels.shape[1]) // 2
        padded_voxels[
            :,
            pad_width : pad_width + voxels.shape[1],
            pad_width : pad_width + voxels.shape[2],
            pad_width : pad_width + voxels.shape[3],
        ] = voxels
        voxels = Volume(padded_voxels)
        voxels.save(os.path.join(folder_name, "gt_vols.mrc"), overwrite=True)
    else:
        voxels = readVols(vols, in_list=False)

    sim = SimulatedSource(n, vols=voxels, unique_filters=filters, noise_var=0)
    var = torch.var(sim._clean_images).item()

    vectorsGT = volsCovarEigenvec(voxels)
    snr_vals = 10 ** np.arange(0, -3.5, -0.5)
    objs = ["ml", "ls"]
    for snr in snr_vals:
        noise_var = var / snr
        logger.info(f"Signal power : {var}. Using noise variance of {noise_var} to achieve SNR of {snr}")

        sim.noise_var = noise_var
        noise_var = sim.noise_var
        dataset = CovarDataset(
            sim,
            noise_var,
            mean_volume=Volume(voxels.asnumpy().mean(axis=0)),
            mask=Volume.load(mask) if mask is not None else None,
        )
        gt_data = GTData(vectorsGT)

        for obj in objs:
            dir_name = os.path.join(folder_name, f"obj_{obj}", f"algorithm_output_{snr}")
            os.makedirs(dir_name, exist_ok=True)
            if save_source:
                sim.save(dir_name)
            display_source(sim, os.path.join(dir_name, "clean_images.jpg"), display_clean=True)
            display_source(sim, os.path.join(dir_name, "noisy_images.jpg"), display_clean=False)
            data_dict, _, _ = covar_processing(
                dataset, r, dir_name, gt_data=gt_data, max_epochs=20, objective_func=obj, num_reg_update_iters=1
            )

            coords_est = data_dict["coords_est"]
            state_centers = np.zeros((len(voxels), coords_est.shape[1]))
            for i in range(len(voxels)):
                state_centers[i] = coords_est[sim.states == i].mean(axis=0)
            with open(os.path.join(dir_name, "state_centers.pkl"), "wb") as f:
                pickle.dump(state_centers, f)

            analyze(
                os.path.join(dir_name, "recorded_data.pkl"),
                output_dir=dir_name,
                analyze_with_gt=True,
                skip_reconstruction=True,
                gt_labels=sim.states,
                latent_coords=os.path.join(dir_name, "state_centers.pkl"),
            )


def simulate_noisy_rots(
    folder_name,
    snr=None,
    noise_var=None,
    rots_std=0,
    offsets_std=0,
    L=64,
    r=5,
    n=100000,
    no_ctf=False,
    vols=None,
    mask=None,
):
    os.makedirs(folder_name, exist_ok=True)

    if vols is None:
        pixel_size = 3 * 128 / L
        voxels = LegacyVolume(L=int(L * 0.7), C=r + 1, K=64, dtype=np.float32, pixel_size=pixel_size).generate()
        padded_voxels = np.zeros((r + 1, L, L, L), dtype=np.float32)
        pad_width = (L - voxels.shape[1]) // 2
        padded_voxels[
            :,
            pad_width : pad_width + voxels.shape[1],
            pad_width : pad_width + voxels.shape[2],
            pad_width : pad_width + voxels.shape[3],
        ] = voxels
        voxels = Volume(padded_voxels)
        voxels.save(os.path.join(folder_name, "gt_vols.mrc"), overwrite=True)
        pixel_size = 3 * 128 / L
    else:
        voxels = Volume.load(vols) if isinstance(vols, str) else readVols(vols, in_list=False)
        if voxels.resolution > L:
            voxels = voxels.downsample(L)
        pixel_size = voxels.pixel_size

    if not no_ctf:
        # filters = [RadialCTFFilter(defocus=d,pixel_size=pixel_size) for d in np.linspace(8e3, 2.5e4, 927)]
        filters = [
            RadialCTFFilter(defocus=d, pixel_size=pixel_size)
            for d in np.random.lognormal(np.log(20000), 0.3, size=(928))
        ]
    else:
        filters = [ArrayFilter(np.ones((L, L)))]

    sim = SimulatedSource(
        n, vols=voxels, unique_filters=filters, noise_var=0, rotations_std=rots_std, offsets_std=offsets_std
    )
    var = torch.var(sim._clean_images).item()

    assert (snr is None) + (noise_var is None) == 1
    if noise_var is None:
        noise_var = var / snr
    sim.noise_var = noise_var
    noise_var = sim.noise_var

    # Place mean est and class vols in the output dir
    output_dir = os.path.join(folder_name, "result_data")
    os.makedirs(output_dir, exist_ok=True)
    mean = voxels.asnumpy().mean(axis=0)
    Volume(mean, pixel_size=pixel_size).save(os.path.join(output_dir, "mean_est.mrc"), overwrite=True)
    voxels.save(os.path.join(output_dir, "class_vols.mrc"), overwrite=True)
    vectorsGT = volsCovarEigenvec(voxels)
    dataset = CovarDataset(sim, noise_var, mean_volume=mean, mask=load_mask(mask, L), apply_preprocessing=False)

    gt_data = GTData(vectorsGT, mean, sim._rotations, sim._offsets)

    sim.save(folder_name, gt_pose=False)
    os.makedirs(os.path.join(folder_name, "gt"), exist_ok=True)
    sim.save(os.path.join(folder_name, "gt"), save_image_stack=False, gt_pose=True)
    display_source(sim, os.path.join(folder_name, "clean_images.jpg"), display_clean=True)
    display_source(sim, os.path.join(folder_name, "noisy_images.jpg"), display_clean=False)
    with open(os.path.join(output_dir, "dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)
    with open(os.path.join(output_dir, "gt_data.pkl"), "wb") as f:
        pickle.dump(gt_data, f)


if __name__ == "__main__":
    ribo_vols = [
        os.path.join("data/scratch_data/cryodrgn_ribosomes/ribosomes/inputs", v)
        for v in os.listdir("data/scratch_data/cryodrgn_ribosomes/ribosomes/inputs")
        if v.endswith(".mrc")
    ]
    simulateExp(
        "data/scratch_data/cryodrgn_ribosomes/ribosomes",
        save_source=False,
        vols=ribo_vols,
        mask="data/scratch_data/cryodrgn_ribosomes/ribosomes/ribo_mask.mrc",
        L=128,
    )
    # [f'data/scratch_data/igg_1d/vols/128_org/{i:03}.mrc' for i in range(0,100,10)]
    # simulate_noisy_rots('data/pose_opt_exp_offsets_snr0.1',snr=0.1,rots_std = 0.1,offsets_std=0.008,r=5,
    #   vols = [f'data/scratch_data/igg_1d/vols/128_org/{int(i):03}.mrc' for i in np.linspace(0,100,6,endpoint=False)],
    #   mask='data/scratch_data/igg_1d/init_mask/mask.mrc')
