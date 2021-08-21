
# Instructions for Generating plots

1. Install dependencies
```
cd plots
pip install -r requirements.txt
cd libraries/qparse
pip install .
cd ..
cd libraries/fplt
pip install .
cd ..
```

2. Download Wandb summary file
```
python data_helpers.py --summary
```

3. To generate plots, run desired `.sh` files
