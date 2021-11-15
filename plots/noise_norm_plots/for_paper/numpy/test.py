import torch
import numpy as np
import math
import glob


def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N / m)  # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm() + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff


for np_name in glob.glob("*.np[yz]"):
    norm = np.load(np_name)
    print(np_name)
    norm = torch.from_numpy(norm[:-1])
    N = len(norm)
    for i in range(1, 1 + int(math.sqrt(N))):
        if N % i == 0:
            m = i
    alpha = alpha_estimator(m, norm)

    print("ALPHA", alpha)
