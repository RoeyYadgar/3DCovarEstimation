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
    
    def get_offsets(self):
        return self.offsets.weight.data

        