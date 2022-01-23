import torch
import numpy as np
import os
import shutil
import logging


def Cutout_batch(input, v, num_k): # length = v
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    output = torch.zeros_like(input).cuda()
    batch_size = input.size(0)//num_k
    for idx in range(batch_size):
        img = input[idx]
        length = v
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).cuda()
        mask = mask.expand_as(img)
        for c in range(num_k):
            img = input[idx + (c-1)*batch_size]
            img *= mask
            output[idx + (c-1)*batch_size] = img
    return output


def Cutmix_batch(x, y, num_k, beta=1.0, cutmix_prob=0.5):
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    r = np.random.rand(1)
    batch_size = x.size(0)//num_k
    if beta > 0 and r < cutmix_prob:
        lam = np.random.beta(beta, beta)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

        for c in range(num_k):
            rand_index_base = torch.randperm(x.size()[0]//num_k).cuda()
            if c == 0:
                rand_index = rand_index_base
            else:
                rand_index = torch.cat([rand_index, (rand_index_base+c*batch_size)], dim=0)

        target_a = y
        target_b = y[rand_index]
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        # compute output
        y = target_a * lam + target_b * (1. - lam)
    return x, y

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + '/'+'model_best.pth.tar')

def copy_checkpoint(top1, top5, directory, filename='model_best.pth.tar'):
    """copy checkpoint"""
    filename = directory + filename
    shutil.copyfile(filename, directory + '/'+str(top1)+'_'+str(top5)+'.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_logger(log_folder, modname=__name__):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_folder,
                        filemode='w')
    logger = logging.getLogger(modname)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger.propagate = False
