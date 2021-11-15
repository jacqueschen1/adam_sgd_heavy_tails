import torch
import pandas as pd
import numpy as np
import urllib
from torch.utils.data import Dataset, DataLoader


class AbaloneDataset(Dataset):
    def __init__(self, x, y):
        # print("y shape", y.shape)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]

    def __len__(self):
        return self.y.shape[0]


def abalone_loader(batch_size, workspace_dir):

    # CSV version of the abalone dataset downloaded from https://datahub.io/machine-learning/abalone/r/abalone.csv
    # We predict the age, which is 1.5 + rings value for each sample
    file_path = workspace_dir + "/datasets/abalone.csv"
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        url = "https://datahub.io/machine-learning/abalone/r/abalone.csv"
        urllib.request.urlretrieve(url, file_path)
        data = pd.read_csv(file_path)

    columns = [column.lower() for column in data.columns.values]

    # Get the index of rings in the data
    index = [i for i, column in enumerate(columns) if column == "class_number_of_rings"]
    index = index[0]

    # For the sex column, M = 0, F = 1, and I = -1
    sex = data.iloc[:, 0].to_numpy()
    m, f, i = sex == "M", sex == "F", sex == "I"
    sex[m], sex[f], sex[i] = 0.0, 1.0, -1.0
    sex_vector = sex.reshape(len(sex), 1)

    # For the rings column, add 1.5 to get age
    age = (data.iloc[:, 8] + 1.5).to_numpy()
    age_vector = age.reshape(len(age), 1)

    X = np.hstack([sex_vector, data.iloc[:, 1:8].to_numpy()])
    y = age_vector

    # Split the data into the training and testing subsets
    train_percent = 0.8
    split = int(round(len(X) * train_percent))
    x_train, x_test = X[:split, :], X[split:, :]
    y_train, y_test = y[:split], y[split:]

    # print(x_train)
    # print(y_train)

    train_dataset = AbaloneDataset(x_train, y_train)
    valid_dataset = AbaloneDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader
