# X-Recon: Learning based Patient-specific High-Resolution CT Reconstruction from Orthogonal X-Ray Projections

## Repository content
This is the pre-release code of the X-Recon: Learning-based Patient-specific High-Resolution CT Reconstruction from Orthogonal X-Ray Images. The original paper presents a CT ultra-sparse reconstruction network based on ortho-lateral chest X-ray images. In this repository, we provide inference codes and pre-trained network weights for automatically reconstructing CT images from ortho-lateral chest X-ray images. Everyone can easily generate the CT images through simple configuration. In addition, we provide a small anonymous demo dataset for reference.

## Install dependencies

Related environment and packages:
```
Python >= 3.6
torch==1.10.0+cu113
numpy
SimpleITK
```
We highly recommend using conda to do this.
```
conda create -n recon python=3.6
conda activate recon
pip install -r requirements.txt
```

## Code Structure
The overall code structure is shown below.
```
├── README.md
├── data: # All the example data is stored in this folder.
│   ├── demo
│   │   ├── filelist.txt
│   │   └── h5-files
│   └── split-dataset.py
├── experiment # The example configuration file is stored in this folder.
│   └── multiview
│       └── ptx_multiview.yml
├── infer.py # 
├── lib
│   ├── __init__.py
│   ├── config # The configuration class is defined in this folder.
│   │   ├── __init__.py
│   │   └── config.py
│   ├── dataset # The data related classes are defined inside this folder.
│   │   ├── __init__.py
│   │   ├── alignDataSetViews.py
│   │   ├── baseDataSet.py
│   │   ├── collate_fn.py
│   │   ├── data_augmentation.py
│   │   ├── data_augmentation_baseline.py
│   │   ├── factory.py
│   │   └── utils.py
│   ├── model # The details of the model are defined inside this folder.
│   │   ├── ProSTModule
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── factory.py
│   │   ├── loss
│   │   ├── multiView_CTGAN.py
│   │   └── nets
│   └── utils # Some useful tools.
│       ├── __init__.py
│       ├── ct.py
│       ├── evaluate_util.py
│       ├── html.py
│       ├── image_pool.py
│       ├── image_utils.py
│       ├── metrics.py
│       ├── metrics_np.py
│       ├── ssim.py
│       ├── transform_3d.py
│       └── visualizer.py
├── requirements.txt # Dependency libraries that are necessary for the program to run.
├── result
│   └── ptx
│       └── demo_test_200
└── save_models
    └── multiView_CTGAN
        └── ptx
```

## Usage
- Firstly, you can download our demo dataset and pre-trained model from: <a href="https://drive.google.com/file/d/1I0sZY73jo8FQSHzACZuJL3X0uv1rFgjX/view?usp=sharing">Google Drive</a> or <a href="https://pan.baidu.com/s/1wfnk2Rh4xLhdYG_FypB3cw?pwd=88zx">Baidu Netdisk</a>. By the way, if you don't have enough computational resources, the example inference results can also be downloaded from the link above.
- Second, extract `data` and `save_models` to their corresponding locations according to the code structure mentioned above.
- Third, execute the script for the inference step. The whole inference requires at least **13G** of VRAM.
```
CUDA_VISIBLE_DEVICES=0 python infer.py \
--ymlpath=./experiment/multiview/ptx_multiview.yml \
--gpu=0 \
--dataroot=./data/demo/h5-files \
--dataset=test \
--tag=demo \
--data=ptx \
--dataset_class=newaug_align_ct_xray_views_std \
--model_class=MultiViewCTGAN \
--datasetfile=./data/demo/filelist.txt \
--resultdir=./result \
--check_point=200 \
--how_many=10 \
--which_model_netG=multiview_network_denseUNetFuse_transposed_skipconnect
```
- Finally, check the `result` folder for the real CT and the reconstructed CT.
## Result
The following image shows the real CT (left) as well as a reconstructed CT (right) of a typical patient with a pneumothorax.

![demo-result](https://github.com/wangyunpengbio/X-Recon/raw/main/imgs/demo-result-high.gif)

## TODO
The training code will be uploaded after the article is accepted.

## Acknowledgement
We thank the Shanghai Public Health Clinical Center for providing the data. We also thank <a href="https://github.com/kylekma">Ma, Kai</a>, and others for providing the open-source code that served as the baseline for this project. This work was also supported by the Peak Disciplines (Type IV) of Institutions of Higher Learning in Shanghai and the Medical Science Data Center of Fudan University.