# $\boldsymbol{\mathrm{DM2F^{+}-GAN}}$
Rui Lin(Github:JacklE0niden)

baseline:DM2F By Zijun Deng, Lei Zhu, Xiaowei Hu, Chi-Wing Fu, Xuemiao Xu, Qing Zhang, Jing Qin, and Pheng-Ann Heng.

This repo is the implementation of
"[Deep Multi-Model Fusion for Single-Image Dehazing](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.pdf)"
(ICCV 2019), written by Zijun Deng at the South China University of Technology.

## Results

Some basic dehazing results can be found at `Report.md` 


* Download model weight files：https://pan.baidu.com/s/1gnxMVDx2S6QHSJdwG9FX0Q?pwd=8nfj


## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Preprocess data
1. Run by ```python tools/preprocess_ohaze_data.py```


## Training

1. ~~Set the path of pretrained ResNeXt model in resnext/config.py~~
2. Set the path of datasets in tools/config.py
3. Run by ```train.sh```

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run by ```test.sh```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

## directory
-DM2F
   -dataset.py
   -train_1_baseline.py
   -data
      -O-HAZE
      -# O-HAZY NTIRE 2018
         -GT
         -HAZY 