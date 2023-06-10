# AGSS
This is the demo code for ''Adapting Generic RGB-D Salient Object Detection for Specific Scenarios''.

## Getting Started
### Requirements
* Python 3.7+
* Pytorch 1.6.0+
* CUDA v10.1, cudnn v.7.5.0
* torchvision

### Data Preprocessing
Download the training (NJUD and NLPR) and testing datasets.

### Test 
```python3 test.py```

## Citation
Please cite the following article when referring to this method.
```
@ARTICLE{9894275,
  author={Song, Mengke and Song, Wenfeng and Yang, Guowei and Chen, Chenglizhao},
  journal={IEEE Transactions on Image Processing}, 
  title={Improving RGB-D Salient Object Detection via Modality-Aware Decoder}, 
  year={2022},
  volume={31},
  number={},
  pages={6124-6138},
  doi={10.1109/TIP.2022.3205747}}
```

## Acknowledgement 
Thanks to [BBSNet](https://github.com/zyjwuyan/BBS-Net) and [GloRe](https://github.com/facebookresearch/GloRe).
