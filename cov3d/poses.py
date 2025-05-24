import torch
import numpy as np
from aspire.volume import rotated_grids
from aspire.utils import grid_2d


def pose_cryoDRGN2APIRE(poses,L):
    rots = np.transpose(poses[0],axes=(0,2,1))
    offsets = poses[1] * L

    return rots, offsets

def pose_ASPIRE2cryoDRGN(rots,offsets,L):
    rots = np.transpose(rots,axes=(0,2,1))
    offsets = offsets / L

    return (rots, offsets)

def rodrigues_rotation_matrix(rotvecs):
    theta = torch.norm(rotvecs, dim=-1, keepdim=True)  # (N, 1)
    k = rotvecs / (theta + 1e-6)  # Normalize, avoiding division by zero
    
    K = torch.zeros((rotvecs.shape[0], 3, 3), device=rotvecs.device,dtype=rotvecs.dtype)  # (N, 3, 3)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    eye = torch.eye(3, device=rotvecs.device,dtype=rotvecs.dtype).unsqueeze(0)  # (1, 3, 3)
    R = eye + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)
    
    return R

class PoseModule(torch.nn.Module):
    def __init__(self, init_rotvecs,offsets, resolution, dtype=torch.float32):
        super().__init__()
        self.resolution = resolution
        self.dtype = dtype
        # convert to tensor if it's not already
        init_rotvecs = torch.as_tensor(init_rotvecs, dtype=dtype)
        offsets = torch.as_tensor(offsets, dtype=dtype)

        assert init_rotvecs.shape[1] == 3, "Rotation vectors should be of shape (N, 3)"
        assert offsets.shape[1] == 2, "Offsets should be of shape (N, 2)"
        assert init_rotvecs.shape[0] == offsets.shape[0], "Rotation vectors and offsets should have the same number of elements"

        n = init_rotvecs.shape[0]
        self.rotvec = torch.nn.Embedding(num_embeddings=n, embedding_dim=3, sparse=True)
        self.rotvec.weight.data.copy_(init_rotvecs)
        self.offsets = torch.nn.Embedding(num_embeddings=n, embedding_dim=2, sparse=True,
                                          _weight=offsets)

        self._init_grid(dtype)

    def _init_grid(self, dtype):
        grid2d = grid_2d(self.resolution, indexing="yx")
        num_pts = self.resolution**2

        grid = np.pi * np.vstack([
            grid2d["x"].flatten(),
            grid2d["y"].flatten(),
            np.zeros(num_pts, dtype=np.float32),
        ])
        self.xy_rot_grid = torch.tensor(grid, dtype=dtype)


        grid_shifted = torch.ceil(torch.arange(-self.resolution / 2, self.resolution / 2, dtype=self.dtype))
        grid_1d = grid_shifted * 2 * torch.pi / self.resolution
        self.phase_shift_grid_x,self.phase_shift_grid_y = torch.meshgrid(
            grid_1d, grid_1d, indexing="xy"
        )

    def forward(self, index):
        rot_mat = rodrigues_rotation_matrix(self.rotvec(index))
        pts_rot = torch.flip(torch.matmul(
            rot_mat.reshape(len(index)*3, 3),
            self.xy_rot_grid
        ).reshape(len(index), 3, self.resolution**2), dims=[1])
        pts_rot = pts_rot = (torch.remainder(pts_rot + torch.pi , 2 * torch.pi) - torch.pi) #After rotating the grids some of the points can be outside the [-pi , pi]^3 cube

        offsets = -self.offsets(index)
        phase_shift = torch.exp(
            1j * (self.phase_shift_grid_x.unsqueeze(0) * offsets[:, 0].reshape(-1,1,1) +
                  self.phase_shift_grid_y.unsqueeze(0) * offsets[:, 1].reshape(-1,1,1))
        )

        return pts_rot, phase_shift
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.xy_rot_grid = self.xy_rot_grid.to(*args, **kwargs)
        self.phase_shift_grid_x = self.phase_shift_grid_x.to(*args, **kwargs)
        self.phase_shift_grid_y = self.phase_shift_grid_y.to(*args, **kwargs)
        return self
    
    def get_rotvecs(self):
        return self.rotvec.weight.data
    
    def set_rotvecs(self, rotvecs,idx = None):
        if idx is None:
            self.rotvec.weight.data.copy_(rotvecs)
        else:
            with torch.no_grad():
                self.rotvec.weight[idx] = rotvecs
    
    def get_offsets(self):
        return self.offsets.weight.data
    
    def set_offsets(self, offsets,idx = None):
        if idx is None:
            self.offsets.weight.data.copy_(offsets)
        else:
            with torch.no_grad():
                self.offsets.weight[idx] = offsets


    def split_module(self, permutation=None):
        '''
        Returns two modules, each with non-overlapping subsets of pose entries.
        '''
        n = self.offsets.weight.shape[0]
        device = self.offsets.weight.device
        dtype = self.offsets.weight.dtype

        if permutation is None:
            permutation = torch.arange(n)
        perm = permutation[:n//2], permutation[n//2:]


        # First module: entries at idx
        rotvecs1 = self.rotvec.weight.data[perm[0]].detach().clone()
        offsets1 = self.offsets.weight.data[perm[0]].detach().clone()
        # Second module: entries not in idx
        rotvecs2 = self.rotvec.weight.data[perm[1]].detach().clone()
        offsets2 = self.offsets.weight.data[perm[1]].detach().clone()

        module1 = PoseModule(rotvecs1, offsets1, self.resolution, dtype=dtype)
        module2 = PoseModule(rotvecs2, offsets2, self.resolution, dtype=dtype)

        # Move to same device as original
        module1 = module1.to(device)
        module2 = module2.to(device)

        return module1, module2


    @staticmethod
    def merge_modules(module1, module2, permutation):
        """
        Merges two PoseModule instances into a new PoseModule containing all poses,
        reordered according to the given permutation.
        """
        device = module1.rotvec.weight.device
        dtype = module1.rotvec.weight.dtype
        resolution = module1.resolution

        # Concatenate the weights from both modules
        rotvecs = torch.cat([module1.rotvec.weight.data.detach(), module2.rotvec.weight.data.detach()], dim=0)
        offsets = torch.cat([module1.offsets.weight.data.detach(), module2.offsets.weight.data.detach()], dim=0)

        # Reorder according to permutation
        rotvecs[permutation] = rotvecs.clone()
        offsets[permutation] = offsets.clone()

        merged_module = PoseModule(rotvecs, offsets, resolution, dtype=dtype)
        merged_module = merged_module.to(device)
        return merged_module
            


def estimate_image_offsets(images, reference, upsampling=4, device=None,mask=None):

    if device is not None:
        images = images.to(device)
        reference = reference.to(device)

    n, h, w = images.shape

    images = images*mask if mask is not None else images
    # Cross-correlation via FFT
    f_img = torch.fft.fft2(images,s=(h*upsampling, w*upsampling))
    f_ref = torch.conj(torch.fft.fft2(reference, s=(h*upsampling, w*upsampling)))
    corr = torch.fft.ifft2(f_img * f_ref).real

    # Center the correlation output
    corr = torch.fft.fftshift(corr)

    max_idx = torch.argmax(corr.reshape(n,-1),dim=1)
    max_idx = torch.unravel_index(max_idx, (h*upsampling, w*upsampling))
    shift_y = max_idx[0] - (h*upsampling) // 2
    shift_x = max_idx[1] - (w*upsampling) // 2


    offsets = torch.vstack([shift_y,shift_x]).T / upsampling

    return offsets

def out_of_plane_rot_error(rot1, rot2):
    """
    #Implementation is used from DRGN-AI https://github.com/ml-struct-bio/drgnai/blob/d45341d1f3411d6db6da6f557207f10efd16da17/src/metrics.py#L134
    """
    unitvec_gt = torch.tensor([0, 0, 1], dtype=torch.float32).reshape(3, 1)

    out_of_planes_1 = torch.sum(rot1 * unitvec_gt, dim=-2)
    out_of_planes_1 = out_of_planes_1.numpy()
    out_of_planes_1 /= np.linalg.norm(out_of_planes_1, axis=-1, keepdims=True)

    out_of_planes_2 = torch.sum(rot2 * unitvec_gt, dim=-2)
    out_of_planes_2 = out_of_planes_2.numpy()
    out_of_planes_2 /= np.linalg.norm(out_of_planes_2, axis=-1, keepdims=True)

    angles = np.arccos(np.clip(np.sum(out_of_planes_1 * out_of_planes_2, -1), -1.0, 1.0)) * 180.0 / np.pi

    return angles, np.mean(angles), np.median(angles)