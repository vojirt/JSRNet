# Road Anomaly Detection by Partial Image Reconstruction with Segmentation Coupling 
Pytorch implementation of our ICCV 2021 paper with the pre-trained model used to generate the results presented in the publication.

**Follow up (new) version of this method is available [HERE](https://github.com/vojirt/DaCUP)**

**[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Vojir_Road_Anomaly_Detection_by_Partial_Image_Reconstruction_With_Segmentation_Coupling_ICCV_2021_paper.pdf)
| [Supplementary
Video](https://cmp.felk.cvut.cz/~vojirtom/data/ICCV2021_Supplementary.mp4)** 

If you use this work please cite:
```latex
@InProceedings{Vojir_2021_ICCV,
    author    = {Vojir, Tomas and \v{S}ipka, Tom\'a\v{s} and Aljundi, Rahaf and Chumerin, Nikolay and Reino, Daniel Olmeda and Matas, Jiri},
    title     = {Road Anomaly Detection by Partial Image Reconstruction With Segmentation Coupling},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15651-15660}
}
```

## Overview

The method consists of two main components:
1. A semantic segmentation network. Currently, a DeepLab v3 architecture is
   used (code adopted from
   [jfzhang95](https://github.com/jfzhang95/pytorch-deeplab-xception)
   distributed under MIT licence). The backbone is ResNet-101 pre-trained on
   ImageNet. It was trained on CityScapes dataset and the network weights are
   fixed.
2. An anomaly estimation network. It is a standalone module that uses the
   features extracted from the ResNet-101 backbone and the output of the
   segmentation network before softmax normalization.

The configuration of the network architecture, as published in ICCV2021, is defined in the
configuration file `parameters.yaml`.  The specific model is loaded
dynamically based on its string name (see `MODEL.NET` variable). The same
applies for the loss. The network implementation is located in
`./net/models.py`. 

**DISCLAIMER: This is a research code. There is lot of unused and cluttered code. You running this code means you will not blame the 
author(s) if this breaks your stuff. This code is provided AS IS without warranty of any kind.**

## Training

All configurations of training is done through the configuration file
`./config/defauls.py` or saved configurations of particular network
configurations. To re-create the training of the proposed architecture use the
`parameters.yaml` as a configuration file. Change the training/val/testing data
sources if needed.

The training datasets are set in the `DATASET.TRAIN` and `DATASET.VAL`. They
are a string variables and currently can be from this list
`['cityscapes_2class']` for training
and  `['cityscapes_2class', 'LaF']` for validation.

### Running training script
The training can be run on specific GPU as (using default configuration from `./config/defauls.py`)
```sh
CUDA_VISIBLE_DEVICES=<GPU_ID> python3 train.py
```
or using custom settings, e.g. saved from custom experiment:
```sh
CUDA_VISIBLE_DEVICES=<GPU_ID> python3 train.py --exp_cfg="./path/to/config_file.yaml"
```

### Dataset

Currently only three labels are used, label 0 for anomaly, label 1 for road and
255 for void. The dataloaders needs to provide the gt segmentation using these
labels only. For example, see e.g.
`./dataloaders/datasets/cityscapes_2class.py`.

The datasets loaders are located in `./dataloaders/datasets/`. Each dataset has
its own dataloader class. 

The path to datasets data are stored in `./mypath.py` where the identification
is a string that is then used in the configuration file to set `DATASET.TRAIN` and
in the `./dataloaders/__init__.py` where the dataset are instantiated.

To add new dataset:
1. path to its data needs to be added to `./mypath.py`.  
2. dataloader class needs to be implemented and stored in `./dataloaders/datasets/`.
3. it instantiation needs to be defined in `./dataloaders/__init__.py` 
4. the `DATASET.TRAIN` or `DATASET.VAL` needs to be set to the new dataset name in the configuration file  

## Testing

For the testing, the `./ReconAnon.py` script is used (see the file for
a minimal example). The `exp_dir` parameter needs to be set to point to a root
directory where the `code` directory and `parameters.yaml` are located.  The
inserted path on line 6 in `./ReconAnon.py` need to be set to point to the
`code/config` directory.  The `evaluate` function expect a tensor with size [1, C, H,
W] (i.e. batch size of 1) where the image is normalized into [0,1] range. 

## Models

There are two pre-trained models:
1. The semantic segmentation model (fixed, does not need to be modified).
   The path to the checkpoint `checkpoint-segmentation.pth` needs to be set
   in the configuration file `MODEL.RECONSTRUCTION.SEGM_MODEL` variable.
   Download from
   [gdrive_segmentation_model](https://drive.google.com/file/d/1gLYgTYXpqcNMUtBHTAvTW6GM8-hK7a4p/view?usp=sharing).
2. The model of the anomaly detection network (either train or use pre-trained)
   `<GITREPO/code/checkpoints/>checkpoint-best.pth`. This model used in the
   publication was trained using the parameters provided in the
   `parameters.yaml` configuration file. It used CityScapes datasets for
   training and LaF training data for validation.  The pre-trained model is
   available on request.

## Performance 

The performance is evaluated on the road region using two pixel-wise metrics:
Average Precision (AP) = Area under Precision-Recall Curve, and False Positive
Rate @ 95% True Positive Rate (FPR@95) = False Positive Rate at operating point
where the True Positive Rate is 95%. In the Table the results are shown as AP
/ FPR@95 for each dataset. Note the significant improvement on the "harder"
datasets (RO, RO21).

|                     | LaF        | LaF-train  | FS         | RA         | RO         | OT          |
|---------------------|------------|------------|------------|------------|------------|-------------|
| JSR-Net (ICCV 2021) | 79.4 / 4.3 | 87.8 / 1.7 | 79.3 / 4.7 | 93.4 / 8.9 | 79.8 / 0.9 | 28.1 / 28.7 |


Datasets used for evaluation:
* [0] LaF - Lost and Found dataset Testing split
* [0] LaF-train - Lost and Found dataset Training split (this was used as a validation dataset during training)
* [1] RA - RoadAnomaly
* [2] RO - RoadObstacles
* [3] OT - Obstacle Track 
* [4] FS - FishyScapes dataset (subset of Lost and Found, for backward results comparability)


[0] P. Pinggera, S. Ramos, S. Gehrig, U. Franke, C. Rother, and R. Mester. Lost
and Found: detecting small road hazards for self-driving vehicles. In
International Conference on Intelligent Robots and Systems (IROS), 2016.  

[1] K. Lis, K. Nakka, P. Fua, and M. Salzmann. Detecting the Unexpected via Image
Resynthesis. In Int. Conf. Comput.  Vis., October 2019.

[2] Krzysztof Lis, Sina Honari, Pascal Fua, and Mathieu Salzmann. Detecting
Road Obstacles by Erasing Them, 2020.

[3] [SegmentMeIfYouCan](https://segmentmeifyoucan.com/) benchmark

[4] H. Blum, P. Sarlin, J. Nieto, R. Siegwart, and C. Cadena.  Fishyscapes:
A Benchmark for Safe Semantic Segmentation in Autonomous Driving. In 2019
IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages
2403–2412, 2019.

# Licence
Copyright (c) 2021 Toyota Motor Europe<br>
Patent Pending. All rights reserved.

This work is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc/4.0/)

