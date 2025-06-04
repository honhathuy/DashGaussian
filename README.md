# DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds
| [Webpage](https://dashgaussian.github.io/) | [Paper](https://arxiv.org/pdf/2503.18402) | [arXiv](https://arxiv.org/abs/2503.18402) | [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

The implementation of **DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds**, a powerful 3DGS training acceleration method. Accepted by CVPR 2025 (highlight).

In this repository, we show how to plug DashGaussian into [the up-to-date 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting). 
To notice, the official implementation of 3DGS has been updating since the paper of DashGaussian is published, so the reproduced results from this repository can be different from that reported in the paper.

## Environment Setup
To prepare the environment, 

1. Clone this repository. 
	```
	git clone https://github.com/YouyuChen0207/DashGaussian
	```
2. Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies. 

	Please notice, that the ```diff-gaussian-rasterization``` module contained in this repository has already been switched to the ```3dgs-accel``` branch for efficient backward computation.
3. Install our Lanczos-resampling implementation for downsampling images. 
	```
	pip install -e submodules/FastLanczos
	```

## Run DashGaussian

### Running Comand
Set the data paths in ```scripts/full_eval.sh``` to your local data folder, and run.
```
bash scripts/full_eval.sh
```

### Running Options
In ```full_eval.py```, you can set, 
* ```--dash``` Enable DashGaussian.
* ```--preset_upperbound``` Set the primitive number upperbound manually for DashGaussian and disable the momentum-based primitive upperbound budgeting method. 
* ```--fast``` Using the Sparse Adam optimizer. 

## Results
The following experiment results are produced with a single NVIDIA RTX 3090 GPU.
### Mipnerf-360 Dataset
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 27.48 | 0.8139 | 0.2170 | 2.71M | 20.30 |
| 3DGS-Dash | Adam | 27.39 | 0.8062 | 0.2339 | 2.25M | 10.87 | 
| 3DGS-fast | Sparse Adam | 27.34 | 0.8089 | 0.2262 | 2.42M | 12.55 | 
3DGS-fast-Dash | Sparse Adam | 27.32 | 0.8018 | 0.2415 | 2.06M | 7.29

### Deep-Blending Dataset
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 29.70 | 0.9023 | 0.2408 | 2.46M | 16.37 |
| 3DGS-Dash | Adam | 29.75 | 0.9019 | 0.2511 | 1.79M | 6.49 | 
| 3DGS-fast | Sparse Adam | 29.56 | 0.9068 | 0.2455 | 2.32M | 9.11 | 
3DGS-fast-Dash | Sparse Adam | 29.76 | 0.9021 | 0.2574 | 1.63M | 4.81

### Tanks&Temple Dataset
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 23.85 | 0.8494 | 0.1706 | 1.56M | 12.19 |
| 3DGS-Dash | Adam | 24.09 | 0.8466 | 0.1868 | 1.16M | 6.67 | 
| 3DGS-fast | Sparse Adam | 23.77 | 0.8505 | 0.1734 | 1.53M | 8.23 | 
3DGS-fast-Dash | Sparse Adam | 23.99 | 0.8455 | 0.1883 | 1.14M | 5.37

## Citation
```
@inproceedings{chen2025dashgaussian,
  title     = {DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds},
  author    = {Chen, Youyu and Jiang, Junjun and Jiang, Kui and Tang, Xiao and Li, Zhihao and Liu, Xianming and Nie, Yinyu},
  booktitle = {CVPR},
  year      = {2025}
}
```
