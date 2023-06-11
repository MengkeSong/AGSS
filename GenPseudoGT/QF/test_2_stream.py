import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pdb, os, argparse
from scipy import misc
import cv2
from my_evaluator import Eval_thread
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data_2_stream import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt, unknown = parser.parse_known_args()

model = CPD_ResNet()
model.load_state_dict(torch.load(r'traind_models/CPD_Resnet_2_stream_gen_gt\Res_gen_gt_CPD_first50%_237.pth'))
model.cuda()
model.eval()

# for dataset in test_datasets:
# dataset_path = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\flo_quality_assess\前50%\test/'
# save_path = r'C:\Users\Administrator\Desktop\CPD-master\sal_map\test/'
dataset_path = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\VOT16-20/'
save_path = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\VOT16-20\trian_masks_400/'

image_root = dataset_path +'train_images_1500/'
flo_root = dataset_path +'train_flos_1500/'
gt_root = dataset_path +  'train_flos_1500/'
test_loader = test_dataset(image_root,flo_root, gt_root, opt.testsize)
num_h = 0
num_l = 0
for i in range(test_loader.size):
    image,flo, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    flo = flo.cuda()

    _, res = model(image)
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    _, res_flo = model(flo)
    res_flo = F.upsample(res_flo, size=gt.shape, mode='bilinear', align_corners=False)
    res_flo = res_flo.sigmoid().data.cpu().numpy().squeeze()
    res_flo = (res_flo - res_flo.min()) / (res_flo.max() - res_flo.min() + 1e-8)

    # thread_0 = Eval_thread(res_flo, gt, cuda=True)
    # Sm0 = thread_0.run()
    thread = Eval_thread(res, res_flo, cuda=True)
    Sm = thread.run()
    # mae= np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    # print(thread.run())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # if float(Sm) > float(0.870) and float(Sm0) > float(0.870):
    if float(Sm) > float(0.870):
        num_h = num_h+1
        sal = res*res_flo
    # save_path = save_path + 'xianzhu/'

        cv2.imwrite(save_path+name, sal*255)
        print(save_path + name)

print(num_h)
print(num_h/test_loader.size)
# print(num_l/test_loader.size)