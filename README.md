## Heavy-tailed noise does not explain the gap between SGD and Adam on Transformers

Supplementary code for our paper: Heavy-tailed noise does not explain the gap between SGD and Adam on Transformers

| Folder               |                                                           |
|----------------------|-----------------------------------------------------------|
| `experiment_scripts` | Experiment definitions                                    |
| `explib`             | Main library                                              |
| `job_scripts`        | Shell scripts to generate jobs to run on clusters         |
| `plots`              | Generate plots for the paper                              |

## Setup 
To install the dependencies:
```
$ cd explib
$ make install
// If developing
$ make install-dev
```

## Setting up a workspace and experiments

To generate the JSON experiment config files, use the `experiment_maker_cli`. The easiest way to do this is to write a basic Python file, like the ones in `experiment_scripts`, which contain all of our experiment scripts for the paper.

To generate the JSON config file:
```
$ python path/to/python/file path/to/json/save/location
```

The JSON file names will be the unique hash that identifies that experiment's configs. 

To run an experiment locally:
```
$ python -m explib path/to/config/json path/to/workspace
```
Logs and results will all be saved in the specificed `path/to/workspace` 

If working on Compute Canada, you can use this program to generate the Slurm job scripts:
```
python ${repo_location}/job_scripts/job_creator_batch.py /path/to/json/directory /path/to/workspace --time=x:xx:xx --gpu={GPU type}
```
This will automatically save the Slurm scripts in a `jobs` directory in your workspace, with the same hash file name as the JSON.

The default time limit is 0-03:00 (3 hours) and the default GPU is `p100`. For more information on the Compute Canada:

Running jobs: https://docs.computecanada.ca/wiki/Running_jobs#Cancelling_jobs

GPUs: https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm#Requesting_a_P100_GPU_node_on_Cedar

And finally to submit the job:
```
sbatch /path/to/slurm/file
```

## Logs

Datasets will be saved in `path/to/workspace/datasets`\
Logs will be saved in `path/to/workspace/${dataset_name}/${experiment_hash}/logs`\

Local wandb logs will be saved in `path/to/workspace/${dataset_name}/${experiment_hash}/wandb`

We use wandb to virtually store experiment logs and as an easy way to view and generate plots. The first time running experiments, you will be prompted to login to wandb. To set the project name for your logs, create an `env` file and set the `WANDB_PROJECT` variable (used in `explib/logging`).

Our wandb logs are stored in the `test-runs-adam-sgd` project here: https://wandb.ai/jacqueschen1/test-runs-adam-sgd. When generating plots, all the data is pulled directly from this project.

The only important data that not stored on wandb are the saved models and the gradient norms.

The models are saved under `/workspace/path/{dataset_name}/{experiment_hash}/model`

The gradients and the final gradient norms are saved under `/workspace/path/{dataset_name}/{experiment_hash}/noise`
