### Experiment Plots

This folder contains all the scripts to generate the plots. All data is pulled from wandb.

`noise_norm_plots` contains the data and scripts needed to produce the noise norm histograms. \
`for_paper` contains all the scripts generate all other plots in the paper, pulling data from https://wandb.ai/jacqueschen1/test-runs-adam-sgd. Note that some of the filtering code is specific to our experiments. `nice_plots_paper.py` is used to generate the initial training plots and the full batch plots, while `nice_plots.py` is used to generate the big batch plots.

To download the Wandb summary csv file:
```
python for_paper/data_helpers.py --summary
```

To generate all the plots in `for_paper` (this will take a while the first time when fetching data):
```
python for_paper/make_all_plots.py
```
