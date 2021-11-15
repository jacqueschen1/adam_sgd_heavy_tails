import argparse
import sys
import os
from pathlib import Path


def generate_job(
    command, directory, json_directory, gpu, time, mem, cpus, trained_norms, subdir
):
    dir = "jobs"
    if subdir is not None:
        dir = dir + "/" + subdir
    Path(os.path.join(directory, dir)).mkdir(parents=True, exist_ok=True)

    for sub, dirs, files in os.walk(json_directory):
        for json_file in files:
            ext = os.path.splitext(json_file)[-1].lower()
            if ext != ".json":
                continue
            job_hash = json_file.split("/")[-1].strip(".json")
            file = open(os.path.join(directory, dir, job_hash + ".sh"), "w+")
            file.write("#!/bin/sh \n")
            file.write("#SBATCH --account=def-schmidtm \n")
            file.write("#SBATCH --gres=gpu:{} \n".format(gpu))
            file.write("#SBATCH --mem={} \n".format(mem))
            file.write("#SBATCH --time={} \n".format(time))
            if cpus is not None:
                file.write("#SBATCH --cpus-per-task={} \n".format(cpus))
            file.write(
                command
                + " "
                + os.path.join(json_directory, json_file)
                + " "
                + directory
            )
            if trained_norms:
                file.write(" --trained_norms")
            file.write(" --verbose \n")
            file.write("exit")
            file.close()


def main():
    parser = argparse.ArgumentParser(description="job creator")
    parser.add_argument("json_directory")
    parser.add_argument("directory")
    parser.add_argument("--command", default="python -m explib")
    parser.add_argument("--gpu", default=1)
    parser.add_argument("--time", default="0-03:00")
    parser.add_argument("--cpus", default=None)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--trained_norms", action="store_true")
    parser.add_argument("--subdir", default=None)
    args = parser.parse_args()
    generate_job(
        command=args.command,
        directory=args.directory,
        json_directory=args.json_directory,
        gpu=args.gpu,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        trained_norms=args.trained_norms,
        subdir=args.subdir,
    )


if __name__ == "__main__":
    main()
