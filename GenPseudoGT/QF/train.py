import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader,test_dataset
from utils import clip_gradient, adjust_lr


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=400, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=18, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD_ResNet()
else:
    model = CPD_VGG()
model.load_state_dict(torch.load('CPD-R.pth'))
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

total_params = sum(p.numel() for p in model.parameters())
print('Number of Parameters: {}'.format(total_params))

image_root = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\flo_quality_assess\后50%\train\flo/'
gt_root = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\flo_quality_assess\后50%\train\gt/'
# test_image_root = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\flo_quality_assess\前50%\test\flo/'
# test_gt_root = r'D:\111111111111111\Sook\sook\classification\datasets\BBSNet_dataset\GAN_video\flo_quality_assess\前50%\test\gt/'
train_loader,train_size = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# test_loader = test_dataset(test_image_root, test_gt_root ,opt.trainsize)
total_step = len(train_loader)
print("train size", train_size)
# print("test size", test_loader.size)

CE = torch.nn.BCEWithLogitsLoss()

step=0
best_mae=1
best_epoch=0

if opt.is_ResNet:
    save_path = 'traind_models/CPD_Resnet_last_50%/'
else:
    save_path = 'traind_models/CPD_VGG/'
def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        atts, dets = model(images)
        loss1 = CE(atts, gts)
        loss2 = CE(dets, gts)
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'Res_CPD.pth')

def testing(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            _, res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae=mae
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'CPD_epoch_best.pth')
                print('best epoch:{}'.format(epoch))

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    # testing(test_loader, model, epoch, save_path)
