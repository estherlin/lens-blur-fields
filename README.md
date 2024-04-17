# Learning Lens Blur Fields
Official code release for [Learning Lens Blur Fields](https://arxiv.org/abs/2310.11535). For additional details, please refer to the paper. 

If you use parts of this work, or otherwise take inspiration from it, please considering citing our paper:

```
@misc{lin2023learning,
      title={Learning Lens Blur Fields}, 
      author={Esther Y. H. Lin and 
        Zhecheng Wang and 
        Rebecca Lin and 
        Daniel Miau and 
        Florian Kainz and 
        Jiawen Chen and 
        Xuaner Cecilia Zhang and 
        David B. Lindell and 
        Kiriakos N. Kutulakos},
      year={2023},
      eprint={2310.11535},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Directory Structure

```
blur-fields
  ├── config
  │   ├── iphone12pro-wide.json  // MLP configuration
  ├── checkpoints  
  │   └── // folder for blur field networkcheckpoints
  ├── data
  │   └── // folder for unprocessed and training data
  ├── models  
  │   └── iphone12pro0-wide.pth // iphone 12 pro model 0 in paper
  │   └── iphone12pro1-wide.pth // iphone 12 pro model 1 in paper
  ├── notebooks
  │   └── preprocess_iphone12pro.ipynb  // demo of preprocessing pipeline
  │   └── visualize_blur_field.ipynb  // how to extract psfs after training
  ├── run
  │   └── train_iphone12pro_wide.sh  // for iphone 12 pro wide used in paper
  ├── util
  │   └── utils.py  // misc helper functions 
  │   └── preprocess.py  // helper functions (e.g. homographies, centre detections)
  │   └── generate_random_pattern.py  // generates random binary noise images
  ├── README.md  // <- You Are Here
  ├── requirements.txt  // package requirements
  └── train.py  // training code for 5D blur field
```

## Getting Started

### Requirements

This code requires tiny-cuda-nn, see [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for installation instructions (we used version 1.6). Package requirements as of October 2023 are in `\requirements.txt`.

```bash
# Make a conda environment.
conda create --name psfs python=3.10
conda activate psfs

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Install tiny-cude-nn separately
# Installation as of Oct. 2023, you can also get it from their repo
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Data

#### Downloading iPhone 12 Pro Data

You can download the iPhone 12 pro device 0 wide lens data used in the paper [here](https://drive.google.com/drive/folders/1zf2p2Bj_Jxhq4-smq1AsnEwlRfKJH0qC?usp=sharing). We include:

1.  4x downsampled processed iphone 12 pro wide lens data (1.06GB): can directly be used in colab demos
2.  full resolution processed iphone 12 pro wide lens data (18.3GB)
3.  unprocessed calibration pattern captures (1.75GB): for demoing the preprocessing pipeline

Place all data in the `data/` folder.

#### Preprocessing Your Own Data

See `notebooks/preprocess_data.ipynb` for a demo of how to preprocess data.

### Training

A training example using `train.py` can be found in `run/train_iphone12pro_wide.sh`. 

## Acknowledgements

We thank Wenzheng Chen for providing discussions and the script for generating random noise patterns. 
