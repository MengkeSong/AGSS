# import os
# import time
# import argparse
# from my_evaluator import Eval_thread
# import numpy as np
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


class Eval_thread():
    def __init__(self, pred,gt, cuda):
        self.pred = pred
        self.gt = gt
        # self.method = method
        # self.dataset = dataset
        self.cuda = cuda
        # self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        # start_time = time.time()
        # mae = self.Eval_mae()
        # max_f = self.Eval_fmeasure()
        # max_e = self.Eval_Emeasure()
        s = self.Eval_Smeasure()
        # self.LOG('{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure, {:.4f} S-measure..\n'.format(self.dataset, self.method, mae, max_f, max_e, s))
        # return '[cost:{:.4f}s]{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure, {:.4f} S-measure..'.format(time.time()-start_time, self.dataset, self.method, mae, max_f, max_e, s)
        # return 'S-measure {:.4f}: '.format(s)
        return float(s)

    def Eval_Smeasure(self):
        # print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            # for pred, gt in self.loader:
            if self.cuda:
                pred = trans(self.pred).cuda()
                gt = trans(self.gt).cuda()
            else:
                pred = trans(self.pred)
                gt = trans(self.gt)
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                gt[gt >= 0.5] = 1
                gt[gt < 0.5] = 0
                # print(self._S_object(pred, gt), self._S_region(pred, gt))
                Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            img_num += 1.0
            avg_q += Q.item()
        avg_q /= img_num
        return avg_q

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

# pred = Image.open(r'./pred\DSS\ECSSD\0001.png').convert('L')
# gt = Image.open(r'./gt\ECSSD\0001.png').convert('L')
#
# pred = pred.resize((480,480), Image.BILINEAR)
# gt = gt.resize((480,480), Image.BILINEAR)
#
# img_transform = transforms.Compose([transforms.ToTensor()])
# t_pred = img_transform(pred)
# t_gt = img_transform(gt)
#
# pred = t_pred.cpu().clone().numpy()
# gt = t_gt.cpu().clone().numpy()
# pred = pred.squeeze(0) # 压缩一维
# gt = gt.squeeze(0) # 压缩一维
#
# thread = Eval_thread(pred, gt, cuda=True)
# print(thread.run())