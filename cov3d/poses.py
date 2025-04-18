import torch
import numpy as np
from aspire.volume import rotated_grids
from aspire.utils import grid_2d
#from pytorch3d.transforms import axis_angle_to_matrix

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
    def __init__(self, init_rotvecs, resolution, dtype=torch.float32):
        super().__init__()
        self.resolution = resolution

        # convert init_rotvecs to tensor if it's not already
        init_rotvecs = torch.as_tensor(init_rotvecs, dtype=dtype)
        n, k = init_rotvecs.shape

        self.rotvec = torch.nn.Embedding(num_embeddings=n, embedding_dim=k, sparse=True)
        self.rotvec.weight.data.copy_(init_rotvecs)

        self._init_grid(dtype)

    def _init_grid(self, dtype):
        grid2d = grid_2d(self.resolution, indexing="yx")
        num_pts = self.resolution**2

        grid = np.pi * np.vstack([
            grid2d["x"].flatten(),
            grid2d["y"].flatten(),
            np.zeros(num_pts, dtype=np.float32),
        ])
        self.grid = torch.tensor(grid, dtype=dtype)

    def forward(self, index):
        rot_mat = rodrigues_rotation_matrix(self.rotvec(index))  # same indexing
        return torch.flip(torch.matmul(
            rot_mat.reshape(len(index)*3, 3),
            self.grid
        ).reshape(len(index), 3, self.resolution**2), dims=[1])
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.grid = self.grid.to(*args, **kwargs)
        return self
    
    def get_rotvecs(self):
        return self.rotvec.weight.data

        