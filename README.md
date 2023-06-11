# AGSS
This is the demo code for ''Adapting Generic RGB-D Salient Object Detection for Specific Scenarios''.

## Getting Started
### Requirements
* Python 3.7+
* Pytorch 1.6.0+
* CUDA v10.1, cudnn v.7.5.0
* torchvision

### Data Preprocessing
Download the RGB-D training datasets and testing datasets from [datasets](https://github.com/jiwei0921/RGBD-SOD-datasets).

Download the stereo training datasets and testing datasets from [datasets](https://github.com/jiwei0921/RGBD-SOD-datasets).

### Step Warming up
Generate optical flow maps (refer to ./GenOpticalFlow/)

### Step 1
Generate saliency informative depth (refer to ./GenDepth/)

### Step 2
Generate high-quality pseudo-GT (refer to ./GenPseudoGT/)

### Step 3 
Target Domain Adaption (refer to ./Target Models/)

* Modify your path of testing dataset in test.py
* Run test.py to inference saliency maps
* Saliency maps generated from the three target models can be downnloaded from [here](https://github.com/jiwei0921/RGBD-SOD-datasets). Code: hp32


## Acknowledgement 
Thanks to [SPNet](https://github.com/taozh2017/SPNet), [SSL](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) and [C2DFNet](https://github.com/Zakeiswo/C2DFNet/tree/main).
