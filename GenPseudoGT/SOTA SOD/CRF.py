import pydensecrf.densecrf as dcrf
import numpy as np
import cv2
import os
def crf_refine(img, annos):      #use crf to refine predict pic
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # assert img.dtype == np.uint8
    # assert annos.dtype == np.uint8
    # print(img.shape[:2],annos.shape)
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32') # set a U which is the same size as input pic
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])  # the same size with the input pic
    return res.astype('uint8')

# masks_path = r'F:\Sook\datasets\Video8K\1Enhanced datasets\Autodrive\BDD100K\bdd100k\videos\test_gt/'
# names = os.listdir(masks_path)
# rgb_path = r'F:\Sook\datasets\Video8K\1Enhanced datasets\Autodrive\BDD100K\bdd100k\videos\test_rgb/'
# save_path = r'F:\Sook\datasets\Video8K\1Enhanced datasets\Autodrive\BDD100K\bdd100k\videos\test_rgb_crf/'
masks_path = r'C:\Users\dell\Desktop\test_depth/'
names = os.listdir(masks_path)
rgb_path = r'C:\Users\dell\Desktop\rgb/'
save_path = r'C:\Users\dell\Desktop\out/'

for name in names:
    img = cv2.imread(rgb_path+name[:-4]+'.jpg')
    mask = cv2.imread(masks_path+name, 0)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # img.shape = np.array(img).shape[:2]
    # mask.shape = np.array(mask).shape
    # print('ok')
    # print('img.shape: ',img.shape, ';', 'mask.shape: ',mask.shape )
    prediction = crf_refine(np.array(img), np.array(mask))
    cv2.imwrite(save_path+name, prediction)
    print(save_path+name)