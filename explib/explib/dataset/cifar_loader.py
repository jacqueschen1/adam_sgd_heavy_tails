import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def cifar_loader(batch_size, workspace_dir, load_100=False, drop_last=False):

    data_class = "CIFAR100" if load_100 else "CIFAR10"

    stats = (
        {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}
        if load_100
        else {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
    )

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats),
    ]

    tr_data = getattr(datasets, data_class)(
        root=workspace_dir + "/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(trans),
    )

    te_data = getattr(datasets, data_class)(
        root=workspace_dir + "/datasets",
        train=False,
        download=True,
        transform=transforms.Compose(trans),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=batch_size,
        shuffle=False,
        # drop_last=drop_last,
    )

    return train_loader, val_loader
