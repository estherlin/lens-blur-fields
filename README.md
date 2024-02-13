# blur field code release
Code release for blur fields paper

## Getting started on GPU
### Getting onto GPUs

Step 1: get onto a lab computer (apollo)
```
ssh lin@apollo.dgp.toronto.edu
```

Step 2: get onto a gpu (triton/tyche/belle, etc.)
```
ssh triton.dgpsrv.sandbox
```

### File locations

Where the repo should be stored:
Put everything you generate in the following directory:

```
/scratch/year/lin/release/
```

Here is the conda environment that you can directly use (no need to build requirements.txt yourself):
```
/scratch/ondemand27/lin/envs/psfs/
```

Here’s how to activate when you get onto the cluster:
```
source ~/.bashrc
conda activate /scratch/ondemand27/lin/envs/psfs/
```

### Existing data to use for training
There is already some existing data in the repo directory that we can use for testing. It is located at:
```
/scratch/year/lin/release/data/iphone12pro/
```

### Saving temporary results:
We can save temporary results to disk at:
```
/scratch/year/lin/release/_results/
```

## Setup

To do a test training run, 
```
cd release/
bash run/train_iphone12pro_wide.sh
```

## Preprocessing

### iPhone pipeline
1. Take focal stacks of each pattern. Each focal stack will be saved in a separate folder
2. Edit folder names to correspond to the pattern
3. Upload main experiment directory to blur-fields/iphone12pro<device #>-<lens>
4. Make entry of experiment params in Notion
5. Extract data from individual images into `.npy` form by running `bash run/process_iphone.sh` in `learn-psfs/`. This create a new folder called `processed` in your experiment folder that stores all of the processed results. _IMPORTANT_: Make sure that no pixels are saturated in the captures!!! Warning messages may be printing on the terminal. 


