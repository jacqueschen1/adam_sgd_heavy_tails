import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, USPS


def torchvision_loader(dataset_name, batch_size, workspace_dir, drop_last=False):
    if dataset_name == "mnist":
        loader = MNIST
    elif dataset_name == "usps":
        loader = USPS
    else:
        raise Exception("Dataset {} not available".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        loader(
            workspace_dir + "/datasets/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        loader(
            workspace_dir + "/datasets/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        # drop_last=drop_last,
    )

    return train_dataloader, valid_dataloader
