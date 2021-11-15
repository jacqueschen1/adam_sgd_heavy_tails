import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.random import default_rng
import math


class SimpleLogistic(Dataset):
    def __init__(self, mean_dist, var, num_samples, seed):

        samples_per_dist = num_samples // 2
        rng = default_rng(seed=seed)
        sd = math.sqrt(var)

        gauss_1 = rng.normal(0, sd, samples_per_dist)
        gauss_2 = rng.normal(mean_dist, sd, samples_per_dist)
        self.x = torch.from_numpy(np.concatenate((gauss_1, gauss_2))).float()

        targets_1 = torch.zeros(samples_per_dist)
        targets_2 = torch.ones(samples_per_dist)
        self.y = torch.cat((targets_1, targets_2))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.y.shape[0]


def simple_logistic_loader(batch_size, workspace_dir, dataset_params):

    valid_size = dataset_params.num_samples // 4

    train_dataset = SimpleLogistic(
        dataset_params.mean_dist,
        dataset_params.var,
        dataset_params.num_samples,
        dataset_params.seed,
    )
    valid_dataset = SimpleLogistic(
        dataset_params.mean_dist, dataset_params.var, valid_size, dataset_params.seed
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader
