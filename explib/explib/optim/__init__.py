"""Optimizers

Generic interface to build optimizers by name,
possibly interfacing with pytorch

"""
import torch
from .sls import Sls
from .adasls import AdaSLS
from .signum import Signum

AVAILABLE_OPTIMIZERS = ["SGD", "Adam", "SGD_Armijo", "Adam_Armijo", "Signum"]


def init(params, model, n_batches):
    name = params["name"]
    momentum = params["momentum"] if "momentum" in params else 0

    if name not in AVAILABLE_OPTIMIZERS:
        raise Exception("Optimizer {} not available".format(name))

    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(), lr=params["alpha"], momentum=momentum
        )

    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=params["alpha"],
            betas=(params["b1"], params["b2"]),
        )

    if name == "Signum":
        return Signum(model.parameters(), lr=params["alpha"], momentum=momentum)

    if name == "SGD_Armijo" or name == "Adam_Armijo":
        if "c" in params:
            c = params["c"]
        else:
            c = 0.1
        if name == "SGD_Armijo":
            return Sls(
                model.parameters(),
                c=c,
                init_step_size=1,
                n_batches_per_epoch=n_batches,
                line_search_fn="armijo",
                beta_b=0.9,
                gamma=2.0,
                beta_f=2.0,
                reset_option=1,
                eta_max=10,
                bound_step_size=True,
            )

        if name == "Adam_Armijo":
            return AdaSLS(
                model.parameters(),
                c=c,
                init_step_size=1,
                n_batches_per_epoch=n_batches,
                base_opt="adam",
                line_search_fn="armijo",
                step_size_method="sls",
            )
