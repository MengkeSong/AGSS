# AGSS
This is the demo code for ''Adapting Generic RGB-D Salient Object Detection for Specific Scenarios''.

## Getting Started
### Requirements
* Python 3.7+
* Pytorch 1.6.0+
* CUDA v10.1, cudnn v.7.5.0
* torchvision

### Data Preprocessing
Download the RGB-D training datasets and testing datasets from [dataset1](https://pan.baidu.com/s/1_H8XKuG7eVu-vM1DMTnA-Q?pwd=qj71) Code: qj71 and [dataset2](https://pan.baidu.com/s/1Mln-DGF4NMkuuzIxtvTu9g?pwd=ehh8) Code: ehh8.

Download the stereo training datasets and testing datasets from [datasets](https://pan.baidu.com/s/1ukuVfK51NuxZU_TsqLVS4A?pwd=8nrw) Code: 8nrw.

Download our collected Video8K from [dataset1](https://pan.baidu.com/s/1N0-md71B4ZLUbGfWScQJ2Q?pwd=fvfd) Code: fvfd and [dataset2](https://pan.baidu.com/s/1oK9eNBHW6t-2K4vzVZv_fQ?pwd=rkmr) Code: rkmr.

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
* Saliency maps on 9 RGB-D testing datasets and 2 stereo testing datasets generated from the three target models can be downnloaded from [here](https://pan.baidu.com/s/1lqUZBuPEfZLmOJqIewu1jg?pwd=awc4) Code: awc4.


## Acknowledgement 
Thanks to [SPNet](https://github.com/taozh2017/SPNet), [SSL](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) and [C2DFNet](https://github.com/Zakeiswo/C2DFNet/tree/main).
