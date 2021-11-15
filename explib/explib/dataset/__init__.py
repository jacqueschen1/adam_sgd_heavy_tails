"""Datasets

General interface to load a dataset

"""

from .torchvision_loader import torchvision_loader
from .abalone_loader import abalone_loader
from .imagenet_loader import imagenet_loader_wds, imagenet_loader
from .cifar_loader import cifar_loader
from .language_loader import ptb_loader, wikitext2_loader

# from .wikitext_loader import wikitext_loader
from .squad_loader import squad_loader
from pathlib import Path
import os
from . import config
from . import datasets

AVAILABLE_DATASET = [
    "mnist",
    "abalone",
    "usps",
    "wikitext",
    "wikitext2",
    "imagenet",
    "cifar10",
    "cifar100",
    "ptb",
    "imagenet_not_sharded",
    "squad",
]
AVAILABLE_DATASET.extend(list(config.DSETS.keys()))


def init(
    dataset_name,
    batch_size,
    workspace_dir,
    device,
    model_name,
    model_args=None,
    drop_last=False,
    full_batch=False,
):
    dataset_path = os.path.join(workspace_dir, "datasets")
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    if dataset_name not in AVAILABLE_DATASET:
        raise Exception("Dataset {} not available".format(dataset_name))

    elif dataset_name == "mnist" or dataset_name == "usps":
        return torchvision_loader(
            dataset_name, batch_size, workspace_dir, drop_last=drop_last
        )
    elif dataset_name == "abalone":
        return abalone_loader(batch_size, workspace_dir)
    elif dataset_name == "wikitext" or dataset_name == "wikitext2":
        if model_args is not None and "tgt_len" in model_args:
            tgt_len = model_args["tgt_len"]
        else:
            tgt_len = 150
        return wikitext2_loader(
            batch_size,
            workspace_dir,
            device,
            tgt_len,
            drop_last=drop_last,
            full_batch=full_batch,
        )
    elif dataset_name == "imagenet_not_sharded":
        return imagenet_loader(batch_size, workspace_dir)
    elif dataset_name == "cifar10":
        return cifar_loader(batch_size, workspace_dir, drop_last=drop_last)
    elif dataset_name == "cifar100":
        return cifar_loader(
            batch_size, workspace_dir, load_100=True, drop_last=drop_last
        )
    elif dataset_name == "ptb":
        if model_args is not None and "tgt_len" in model_args:
            tgt_len = model_args["tgt_len"]
        else:
            tgt_len = 128
        return ptb_loader(
            batch_size,
            workspace_dir,
            device,
            tgt_len,
            drop_last=drop_last,
            full_batch=full_batch,
        )
    elif dataset_name == "squad":
        tgt_len = model_args["tgt_len"] if "tgt_len" in model_args else 384
        doc_stride = model_args["doc_stride"] if "doc_stride" in model_args else 128
        return squad_loader(
            batch_size,
            workspace_dir,
            tgt_len,
            doc_stride,
            model_name,
            drop_last=drop_last,
            full_batch=full_batch,
        )
    else:
        return datasets.load(dataset_name, workspace_dir).get_dataloaders(batch_size)
