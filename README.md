# AGSS
This is the demo code for ''Adapting Generic RGB-D Salient Object Detection for Specific Scenarios''.

## Getting Started
### Requirements
* Python 3.7+
* Pytorch 1.6.0+
* CUDA v10.1, cudnn v.7.5.0
* torchvision

### Data Preprocessing
Download the training datasets and testing datasets from [datasets](https://github.com/jiwei0921/RGBD-SOD-datasets).

### Test 
Download pretrained model from here. Code: qcra

* Modify your path of testing dataset in test.py
* Run test.py to inference saliency maps
* Saliency maps generated from the model can be downnloaded from here. Code: hp32
*
```python test.py```


## Acknowledgement 
Thanks to [SPNet](https://github.com/taozh2017/SPNet), [SSL](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) and [C2DFNet](https://github.com/Zakeiswo/C2DFNet/tree/main).
