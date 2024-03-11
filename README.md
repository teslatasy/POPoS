
# PoPoS #
The pytorch implementation for paper " POPoS: Detecting facial landmarks with Parallel Optimal Position Search".

[[Project]](https://teslatasy.github.io/PoPoS/) [[Paper]]() 

## Requirements
- python==3.8.17
- mmcv-full==1.6.2
- mmdet==2.28.2
- mmpose==0.29.0
- openmim==0.3.9
- torch==1.12.0
- torch-summary==1.4.5
- torchaudio==0.12.0
- torchstat==0.0.7
- torchvision==0.13.0
- tornado==6.3.3

For more details, please refer to the `requirements/piplist.txt`. We conduct the experiments with 4 NVIDIA 3090 GPUs.

## Dataset
Please download the 300w dataset for training and test, and process the dataset as following.

**Data Preprocessing:** 
## Quick Start Guide
Get started with the POPoS facial landmark detection system in a few simple steps:

### 1. Installation:

- **Environment Setup**: Begin by setting up the necessary environment. For this, refer to the instructions provided by [mmpose](https://github.com/open-mmlab/mmpose).
  
- **Datasets**: Our experiments utilize the COCO, WFLW, 300W, COFW, and AFLW datasets.

### 2. Training:

- **Pre-trained Models**: We leverage ImageNet models from [mmpose](https://github.com/open-mmlab/mmpose) as our starting point.

- **Training Command**: To start the training process, execute the following command:

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_train.sh \
      configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256_dark.py \
      4 \
      --work-dir exp/exp889
  ```

### 3. Evaluation:

#### Step 1: Obtain the Models
- **Download**: Retrieve the pre-trained and trained models for each dataset and heatmap resolution from [Google Drive]()(coming soon).

#### Step 2: Model Setup
- **Placement**: After downloading, move the "exp" model file to the root directory of your codebase.

#### Step 3: Resolution Configuration
- **Supported Resolutions**: The model in the "exp" directory is compatible with five resolutions: 64, 32, 16, 8, and 4.
  
- **Configuration**: Prior to running the test script, adjust the resolution by editing the "data_cfg/heatmap_size" field in the configuration file to your chosen resolution.

#### Step 4: Test Execution
- **Script Selection**: Based on your chosen resolution, run the appropriate test script:

  - `run_test_4.sh`
  
  These scripts evaluate the model's efficacy across various face datasets: WFLW, COCO, 300W, AFLW, and COFW.

#### Step 5: Evaluation Command
- **Command Execution**: To kick off the evaluation, input the following command:

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
      configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
      exp/exp_v1.3.0/best_NME_epoch_60.pth \
      4 

 


## Acknowledgment
Our work is primarily based on [mmpose](https://github.com/zhiqic/KeyPosS/). We express our gratitude to the authors for their invaluable contributions.
## Citation ##
Please cite the following paper if you use this repository in your research.

```

```