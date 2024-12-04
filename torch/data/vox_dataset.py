import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def reshape_voxels_array(voxels):
    assert len(voxels.shape) == 5

    u, c = np.unique(voxels.shape, return_counts=True)
    assert max(c) == 3
    assert set(c) == set([1, 3])
    assert (np.sort(c) == np.array([1, 1, 3])).all()

    grid_size_indices = list(
        np.where(pd.Series(voxels.shape).duplicated(keep=False))[0]
    )

    samples_dim = [np.argmax(voxels.shape)]
    n_samples = max(voxels.shape)

    channels_dim = list(
        set(range(len(voxels.shape))) - set(grid_size_indices) - set(samples_dim)
    )
    n_channels = voxels.shape[channels_dim[0]]

    grid_size = voxels.shape[grid_size_indices[0]]

    voxels = np.transpose(voxels, tuple(samples_dim + channels_dim + grid_size_indices))

    return voxels, n_samples, n_channels, grid_size


def load_voxels(path_to_voxels, path_to_values):
    full_voxels, n_samples, n_channels, grid_size = reshape_voxels_array(
        np.load(path_to_voxels)
    )

    full_values = np.load(path_to_values)

    return full_voxels, n_samples, n_channels, grid_size, full_values


def random_rotation(polycube):
    def rotations(polycube, i, axes):
        return torch.rot90(polycube, i, axes)

    r = np.random.randint(6)
    rot_degree = np.random.randint(4)

    if r == 0:
        return rotations(polycube, rot_degree, (2, 3))
    elif r == 1:
        return rotations(torch.rot90(polycube, 2, (1, 3)), rot_degree, (2, 3))
    elif r == 2:
        return rotations(torch.rot90(polycube, 1, (1, 3)), rot_degree, (1, 2))
    elif r == 3:
        return rotations(torch.rot90(polycube, -1, (1, 3)), rot_degree, (1, 2))
    elif r == 4:
        return rotations(torch.rot90(polycube, 1, (1, 2)), rot_degree, (1, 3))
    elif r == 5:
        return rotations(torch.rot90(polycube, -1, (1, 2)), rot_degree, (1, 3))


class VoxDataset(Dataset):
    def __init__(
        self,
        voxels: np.ndarray,
        values: np.ndarray,
        n_channels: int,
        grid_size: int,
        device: str = "cuda",
        cubic_rotations: bool = True,
        v_dtype=torch.int64,
    ):
        self.voxels = voxels
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.cubic_rotations = cubic_rotations

        if device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = device

        self.v_dtype = v_dtype
        self.values = values

        self.n_samples = len(self.voxels)

        assert self.n_samples == len(self.values)

    def __len__(self):
        return self.n_samples

    def get_n_channels(self):
        return self.n_channels

    def get_grid_size(self):
        return self.grid_size

    def get_value_dtype(self):
        return self.v_dtype

    def __getitem__(self, i):
        voxel = torch.from_numpy(self.voxels[i, :, :, :, :])

        if self.cubic_rotations:
            voxel = random_rotation(voxel)

        return (
            voxel.to(dtype=torch.float32, device=self.device),
            torch.from_numpy(np.array(self.values[i])).to(
                dtype=self.v_dtype, device=self.device
            ),
        )
