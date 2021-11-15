"""Helpers to create experiments


"""
import hashlib
import json
import os

import argparse


def hash_dict(exp_dict):
    """Create a hash for an experiment"""
    return hashlib.md5(json.dumps(exp_dict, sort_keys=True).encode("utf-8")).hexdigest()


def make_dict_file(folder, exp_dict):
    """Creates an experiment file in the folder"""
    hash = hash_dict(exp_dict)
    filepath = os.path.join(folder, hash + ".json")
    with open(filepath, "w") as fp:
        json.dump(exp_dict, fp)
    return filepath


def experiment_maker_cli(descr, experiments):
    """ """
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print debug information",
    )
    parser.add_argument(
        "path",
        default=".",
        type=str,
        help="Path to the folder where to store ",
    )
    args = parser.parse_args()

    path = args.path

    for exp in experiments:
        filepath = make_dict_file(path, exp)
        if args.verbose:
            print("Created {}".format(filepath))
