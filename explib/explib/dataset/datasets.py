import os
from . import config
import numpy as np
import sklearn as sk
from .downloader import download_and_extract
import pandas as pd
import torch
import scipy as sp
from torch.utils.data import Dataset, DataLoader


class DatasetTorch(Dataset):
    def __init__(self, x, y, task):
        self.x = x.astype(np.float32)
        self.y = torch.from_numpy(y)
        if len(self.y.shape) == 1 and task == config.TASK_REG:
            self.y = self.y.unsqueeze(1)

    def __getitem__(self, idx):

        if sp.sparse.issparse(self.x):
            x_coo = self.x[idx, :].tocoo()
            ind = torch.LongTensor([x_coo.row, x_coo.col])
            data = torch.FloatTensor(x_coo.data)
            return torch.sparse.FloatTensor(ind, data, list(x_coo.shape)), self.y[idx]
        else:
            return torch.from_numpy(self.x[idx, :]).float(), self.y[idx]

    def __len__(self):
        return self.y.shape[0]


class Dataset:
    def __init__(
        self,
        X_tr,
        y_tr,
        X_val=None,
        y_val=None,
        X_te=None,
        y_te=None,
        task=config.TASK_CLASS,
    ):
        assert task in config.AVAILABLE_TASKS
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.X_te = X_te
        self.y_te = y_te
        self.task = task

    def get_train(self):
        return self.X_tr, self.y_tr

    def get_val(self):
        return self.X_val, self.y_val

    def get_test(self):
        return self.X_te, self.y_te

    def __str__(self):
        def get_shape_or_none(x):
            return ("" if x[0] is None else str(x[0].shape)) + (
                "" if x[1] is None else ", " + str(x[1].shape)
            )

        return str(
            {
                "train": get_shape_or_none(self.get_train()),
                "val": get_shape_or_none(self.get_val()),
                "test": get_shape_or_none(self.get_test()),
            }
        )

    def get_dataloaders(self, batch_size):
        x_train, y_train = self.X_tr, self.y_tr
        if self.X_te is None:
            split = int(round(x_train.shape[0] * 0.8))
            x_train, x_test = x_train[:split, :], x_train[split:, :]
            y_train, y_test = y_train[:split], y_train[split:]
        else:
            x_test, y_test = self.X_te, self.y_te

        train_dataset = DatasetTorch(x_train, y_train, self.task)
        valid_dataset = DatasetTorch(x_test, y_test, self.task)

        if sp.sparse.issparse(x_train):
            train_dataloader = DataLoader(
                train_dataset,
                sampler=torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.sampler.RandomSampler(train_dataset),
                    batch_size=batch_size,
                    drop_last=False,
                ),
                batch_size=None,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                sampler=torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.sampler.RandomSampler(valid_dataset),
                    batch_size=batch_size,
                    drop_last=False,
                ),
                batch_size=None,
            )
        else:
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False
            )
        return train_dataloader, valid_dataloader


def is_downloaded(dname, workspace_dir):
    dsinfo = config.DSETS[dname]
    folder_path = os.path.join(workspace_dir, "datasets", dname)
    path_tr = os.path.join(folder_path, dsinfo["train"])
    return os.path.isfile(path_tr)


def load_libsvm(dname, workspace_dir):
    dsinfo = config.DSETS[dname]
    folder_path = os.path.join(workspace_dir, "datasets", dname)

    path_tr = os.path.join(folder_path, dsinfo["train"])
    path_val = os.path.join(folder_path, dsinfo["val"]) if "val" in dsinfo else None
    path_te = os.path.join(folder_path, dsinfo["test"]) if "test" in dsinfo else None

    if not is_downloaded(dname, workspace_dir):
        for url in dsinfo["urls"]:
            download_and_extract(url, folder_path)

    tr = sk.datasets.load_svmlight_file(path_tr)
    # Replace -1 labels with 0
    tr[1][tr[1] == -1] = 0
    tr[1][tr[1] == 2] = 0
    train_set = tr[0]
    if dname != "news20-binary":
        train_set = sp.sparse.csr_matrix.toarray(tr[0])
    val = (
        sk.datasets.load_svmlight_file(path_val)
        if path_val is not None
        else (None, None)
    )
    te = (
        sk.datasets.load_svmlight_file(path_te) if path_te is not None else (None, None)
    )
    test_set = None
    if te[0] is not None:
        print("TE not none")
        te[1][te[1] == -1] = 0
        te[1][te[1] == 2] = 0
        test_set = te[0]
        if dname != "news20-binary":
            test_set = sp.sparse.csr_matrix.toarray(te[0])
        if dname == "a9a":
            feature_col = np.zeros((len(test_set), 1))
            test_set = np.append(test_set, feature_col, axis=1)

    return train_set, tr[1], val[0], val[1], test_set, te[1]


def load_skl(dname):
    loader = getattr(sk.datasets, config.DSETS[dname]["skl_loader"])
    x_tr, y_tr = loader(return_X_y=True)
    return x_tr, y_tr, None, None, None, None


def load_uci(dname, workspace_dir):
    dsinfo = config.DSETS[dname]
    folder_path = os.path.join(workspace_dir, "datasets", dname)

    if not is_downloaded(dname, workspace_dir):
        for url in dsinfo["urls"]:
            download_and_extract(url, folder_path)

    def load_if_exist(data_subset):
        if data_subset not in dsinfo:
            return None, None
        path = os.path.join(folder_path, dsinfo[data_subset])
        if ".xls" in path:
            data = pd.read_excel(path).to_numpy()
        else:
            data = np.loadtxt(path)

        x = data[:, dsinfo["features"]] if "features" in dsinfo else data[:, :-1]
        y = data[:, dsinfo["target"]] if "target" in dsinfo else data[:, -1]

        return x, y

    x_tr, y_tr = load_if_exist("train")
    x_val, y_val = load_if_exist("val")
    x_te, y_te = load_if_exist("test")

    return x_tr, y_tr, x_val, y_val, x_te, y_te


def load(dname, workspace_dir):
    if config.DSETS[dname]["format"] == "skl":
        return Dataset(*load_skl(dname), config.DSETS[dname]["TASK"])
    elif config.DSETS[dname]["format"] == "uci":
        return Dataset(*load_uci(dname, workspace_dir), config.DSETS[dname]["TASK"])
    else:
        return Dataset(*load_libsvm(dname, workspace_dir), config.DSETS[dname]["TASK"])
