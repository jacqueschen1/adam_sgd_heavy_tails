TASK_CLASS = "Classification"
TASK_REG = "Regression"

SIZE_SMALL = "small"
SIZE_MEDIUM = "medium"
SIZE_LARGE = "large"

AVAILABLE_TASKS = [TASK_CLASS, TASK_REG]


def libsvm_url(category, file):
    return "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/{}/{}".format(
        category, file
    )


def libsvm_ds_url(category, dsname):
    return "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/{}.html#{}".format(
        category, dsname
    )


def uci_url(dsname):
    return "https://archive.ics.uci.edu/ml/datasets/{}".format(dsname)


def uci_ds_url(end):
    return "https://archive.ics.uci.edu/ml/machine-learning-databases/{}".format(end)


def skl_url(dsname):
    return "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.{}.html".format(
        dsname
    )


def delve_url(s):
    return "https://www.cs.toronto.edu/~delve/data/{}".format(s)


def delve_ds_url(s):
    return "ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/{}".format(s)


DSETS = {
    "a9a": {
        "url": libsvm_ds_url("binary", "a9a"),
        "urls": [libsvm_url("binary", "a9a"), libsvm_url("binary", "a9a.t")],
        "train": "a9a",
        "test": "a9a.t",
        "format": "libsvm",
        "TASK": TASK_CLASS,
    },
    "covtype-binary": {
        "url": libsvm_ds_url("binary", "covtype.binary"),
        "urls": [libsvm_url("binary", "covtype.libsvm.binary.bz2")],
        "train": "covtype.libsvm.binary",
        "format": "libsvm",
        "TASK": TASK_CLASS,
        "size": SIZE_MEDIUM,
    },
    "covtype-binary-scale": {
        "url": libsvm_ds_url("binary", "covtype.binary"),
        "urls": [libsvm_url("binary", "covtype.libsvm.binary.scale.bz2")],
        "train": "covtype.libsvm.binary.scale",
        "format": "libsvm",
        "TASK": TASK_CLASS,
        "size": SIZE_MEDIUM,
    },
    "news20-binary": {
        "url": libsvm_ds_url("binary", "news20.binary"),
        "urls": [libsvm_url("binary", "news20.binary.bz2")],
        "train": "news20.binary",
        "format": "libsvm",
        "TASK": TASK_CLASS,
    },
    "rcv1-binary": {
        "url": libsvm_ds_url("binary", "rcv1.binary"),
        "urls": [
            libsvm_url("binary", "rcv1_train.binary.bz2"),
            # libsvm_url("binary", "rcv1_test.binary.bz2"),
        ],
        "train": "rcv1_train.binary",
        # "test": "rcv1_test.binary",
        "format": "libsvm",
        "TASK": TASK_CLASS,
        "size": SIZE_MEDIUM,
    },
    "boston-housing": {
        "url": skl_url("load_boston"),
        "skl_loader": "load_boston",
        "format": "skl",
        "TASK": TASK_REG,
    },
    "naval-propulsion": {
        "url": uci_url("Condition+Based+Maintenance+of+Naval+Propulsion+Plants"),
        "urls": [uci_ds_url("00316/UCI%20CBM%20Dataset.zip")],
        "features": slice(None, -2),
        "target": -2,
        "train": "UCI CBM Dataset/data.txt",
        "format": "uci",
        "TASK": TASK_REG,
    },
}
