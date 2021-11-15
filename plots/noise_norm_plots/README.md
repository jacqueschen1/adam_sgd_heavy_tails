## Noise Norm Plots

This folder contains the data and scripts needed to produce the noise norm histograms. All relevant data and scripts to produce the histogram present in the paper are in the `for_paper` folder.

In the `for_paper` folder, to produce the final figure, call
```
python make_hist_paper.py
```

The `numpy` folder contains the actual gradient norms generated from the runs, extracted from the workspace.

As mentioned in the main README:

The only important data that not stored on wandb are the saved models and the gradient norms.

The gradients and the final gradient norms are saved under `/workspace/path/{dataset_name}/{experiment_hash}/noise`
