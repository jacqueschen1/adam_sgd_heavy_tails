import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def imagenet_loader(batch_size, workspace_dir):
    traindir = os.path.join(
        workspace_dir, "datasets", "imagenet", "ILSVRC", "Data", "DET", "train"
    )
    valdir = os.path.join(
        workspace_dir, "datasets", "imagenet", "ILSVRC", "Data", "DET", "val"
    )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=24,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=24,
    )

    print(len(train_loader))
    print(len(val_loader))

    return train_loader, val_loader


def identity(x):
    return x
