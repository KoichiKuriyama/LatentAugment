import argparse
import os
import shutil
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from models.wideresnet import WideResNet
from models.resnet import ResNet
from models.shake_resnet import ShakeResNet
from models.shake_pyramidnet import ShakePyramidNet
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR
from augment.data import get_dataloaders
from augment.policy import LatentPolicy
from augment.utils import Cutout_batch, Cutmix_batch, save_checkpoint, copy_checkpoint, AverageMeter, accuracy, setup_logger
from tqdm import tqdm
from collections import OrderedDict
import logging
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='PyTorch LatentAugment')

# Dataset
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default], cifar100, svhn, imagenet)')
parser.add_argument('--dataroot', default='/path/to/data/', type=str,
                    help='path to data',)
parser.add_argument('--num_workers', type=int, default=4, help='')

# Checkpoints
parser.add_argument('--checkpoint', default='/path/to/checkpoint/', type=str,
                    help='path to checkpoint',)
parser.add_argument('--name', default='LA-cifar10-wrn40x2', type=str,
                    help='name of experiment')

# Parameters of LatentAugment
parser.add_argument('--num_k', default=6, type=int,
                    help='number of augmented data per input data')
parser.add_argument('--transforms', default=16, type=int,
                    help='the number of transforms')
parser.add_argument('--moving', type=int, default=10, help='steps of moving average')
parser.add_argument('--sig', type=float, default=1., help='sigma of h_z')

# Optimization parameters
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
parser.add_argument('--multiplier', default=1, type=int)
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

# model parameters
parser.add_argument('--model', default='wrn40x2', type=str,
                    help='wrn40x2, wrn28x10, resnet50, shakeshake, or pyramid272 (default: wrn40x2)',)
parser.add_argument('--weight-decay', '--wd', default=0.0002, type=float)
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 40)')
parser.add_argument('--widen-factor', default=2, type=int,
                    help='widen factor (default: 2)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--depth', default=272, type=int)
parser.add_argument('--alpha', default=200, type=int)
parser.add_argument('--bottleneck', default=True, type=bool)

# other parameters
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
parser.add_argument('--cutout', default='False', type=strtobool,
                    help='cutout, True: use cutout, False: not use(default)')
parser.add_argument('--cutmix', default='True', type=strtobool,
                    help='cutmix, True: use cutmix(default), False: not use')
parser.add_argument('--disable_tqdm', default='False', type=strtobool,
                    help='disable tqdm, True: disable tqdm, False: use tqdm (default)')

parser.set_defaults(augment=True)
args = parser.parse_args()
#best_prec1 = 0

if not os.path.isdir('./Log'):
    os.makedirs('./Log', exist_ok=True)
log_folder = f"./Log/{args.name}_{datetime.now():%Y%m%d%H%M%S}.log"
logger = setup_logger(log_folder)

N_GPU = torch.cuda.device_count()

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)

def main():
    global args, num_classes
    print("args.epochs")
    print(args.epochs)

    best_prec1 = 0
    best_prec5 = 0
    num_transforms = args.transforms
    res_policy = torch.zeros(args.epochs,num_transforms,num_transforms)
    res_magnitude = torch.zeros(args.epochs,10)

    args = parser.parse_args()
    best_prec1 = 0

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'svhn' or args.dataset == 'svhn_core':
        num_classes = 10
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    elif args.dataset == 'imagenet':
        num_classes = 1000

    train_loader, test_loader, train_dataset, val_dataset, train_sampler = get_dataloaders(
          args.dataset, args.batch_size, args.dataroot, num_classes, args.num_workers,
          args, args.num_k, args.transforms)

    N_Train = len(train_loader.dataset)

    logging.info('ExperimentName: %s' % args.name)
    logging.info('Checkpoint: %s'% args.checkpoint)
    logging.info('DataSource: %s'% args.dataset)
    logging.info(' ')
    if 'wrn' in args.model:
        logging.info('Wide ResNet ')
        logging.info(' Depth: %s' % args.layers)
        logging.info(' Widen_Factor: %s' % args.widen_factor)
        logging.info(' ')
    elif 'resnet' in args.model:
        logging.info('ResNet ')
        logging.info(' Depth: %s' % args.layers)
        logging.info(' ')
    elif 'pyramid' in args.model:
        logging.info('Pyramid ')
        logging.info(' Depth: %s' % args.depth)
        logging.info(' alpha: %s' % args.alpha)
        logging.info(' bottleneck: %s' % args.bottleneck)
    elif 'shakeshake' in args.model:
        logging.info('ShakeShake_ResNet ')
        logging.info(' Depth: %s' % args.layers)
        logging.info(' Widen_Factor:%s' % args.widen_factor)
        logging.info(' ')
    else:
        raise ValueError('invalid model name=%s' % args.model)

    logging.info(' num_k: %s' % args.num_k)
    logging.info(' transforms: %s' % num_transforms)
    logging.info(' ')

    logging.info(' batch_size: %s ' % args.batch_size)
    logging.info(' epoch: %s '% args.epochs)
    logging.info(' weight-decay: %s' % args.weight_decay)
    logging.info(' learning rate: %s' % args.lr)
    logging.info(' ')
    logging.info(' cutout: %s ' % args.cutout)
    logging.info(' cutmix: %s ' % args.cutmix)
    logging.info(' moving average: %s ' % args.moving)
    logging.info(' ')
    logging.info(' N_Train: %s ' % len(train_dataset))
    logging.info(' ')
    logging.info(' N_GPU: %s ' % torch.cuda.device_count())
    logging.info(' seed: %s ' % seed)
    logging.info(' ')
    logging.info(' Sig: %s ' % args.sig)
    logging.info(' ')

    # create model
    if 'imagenet' in args.dataset:
        dataset = 'imagenet'
    elif  'cifar' in args.dataset:
        dataset = 'cifar'
    else:
        dataset = args.dataset

    if 'resnet' in args.model:
        model = ResNet(dataset=dataset, depth=args.layers, num_classes=num_classes, bottleneck=True)
    elif 'wrn' in args.model:
        model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=0)
    elif 'pyramid' in args.model:
        model = ShakePyramidNet(depth=args.depth, alpha=args.alpha, label=num_classes)
    elif 'shakeshake' in args.model:
        if args.dataset == 'cifar10':
            model = ShakeResNet(args.layers, args.widen_factor, num_classes)
        else:
            model = ShakeResNet(args.layers, args.widen_factor, num_classes)

    # get the number of model parameters
    logging.info('Total number of parameters: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    N_GPU = torch.cuda.device_count()
    if N_GPU >= 2:
        model = torch.nn.DataParallel(model)

    cudnn.benchmark = True
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                args.lr,
                momentum=args.momentum,
                nesterov = args.nesterov,
                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.multiplier, total_epoch=args.warmup, after_scheduler=scheduler)


    num_k = args.num_k
    probs = torch.ones(num_transforms,num_transforms) / (num_transforms**2)
    if args.moving > 0:
        n_probs_count = args.moving
    else:
        n_probs_count = len(train_loader)
    probs_count = torch.ones(n_probs_count,num_transforms,num_transforms) / n_probs_count

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            res_policy = checkpoint['res_policy']
            res_magnitude = checkpoint['res_magnitude']

            probs_count = checkpoint['probs_count']
            probs = res_policy[args.start_epoch-1,:,:]

            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            for _ in range(args.start_epoch):
                scheduler.step()
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        ep_time = time.time()
        train_acc, train_acc5, train_loss, probs, probs_count, mag_mix = train(train_loader, model, criterion, optimizer, scheduler, epoch, probs, probs_count, args)
        prec1, prec5, valid_loss = validate(test_loader, model, criterion, epoch)
        ep_time = time.time() - ep_time
        #logging.info('%9.3f' %(ep_time))
        logging.info('epoch=%5d sec=%5.3f lr=%3.5f train_acc1=%3.4f train_acc5=%3.4f test_acc1=%3.4f test_acc5=%3.4f train_loss=%f test_loss=%f'
        %((epoch+1), ep_time, (optimizer.param_groups[0]['lr']), train_acc, train_acc5, prec1, prec5, train_loss, valid_loss))

        res_policy[epoch,:,:]= probs
        res_magnitude[epoch,:]= mag_mix

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)

        directory = args.checkpoint+args.name+'/'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'res_policy': res_policy,
            'res_magnitude' : res_magnitude,
            'best_prec1': best_prec1,
            'probs_count': probs_count,
        }, is_best, directory)

    copy_checkpoint(best_prec1, best_prec5, directory)
    logging.info('Best accuracy: %s', best_prec1)
    logging.info('Best5 accuracy: %s', best_prec5)
    total_time = time.time() - start_time
    logging.info('Total time (sec.): %s',total_time)



def train(train_loader, model, criterion, optimizer, scheduler, epoch, probs, probs_count, args):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mag_cand = torch.zeros(10).cuda()

    model.train()
    with tqdm(train_loader, ncols=150, leave=False, disable=args.disable_tqdm) as pbar:
        pbar.set_description("[train: %d]" % (epoch+1))
        for i_batch, (image_base, y0, x_list, y_list, operation1_z, operation2_z, mag1_z, mag2_z) in enumerate(pbar):
            num_k = args.num_k
            batch_size = image_base.size(0)
            onehot_target = y0
            onehot_target = onehot_target.cuda()
            like_z = torch.zeros(batch_size,num_k).cuda()
            pi_z = torch.zeros(batch_size,num_k).cuda()

            my_seed = np.random.randint(99999999)

            with torch.no_grad():

                torch.manual_seed(my_seed)

                output_base = model(image_base.cuda())
                like_base = torch.sum(F.softmax(output_base, dim=1)*onehot_target,1)

                image_z = x_list.permute(1,0,2,3,4).cuda()
                y_z = y_list.permute(1,0,2).cuda()

                for c in range(num_k):
                    if c == 0:
                        x_all = image_z[c]
                        y_all = y_z[c]
                    else:
                        x_all = torch.cat([x_all,image_z[c]], dim=0)
                        y_all = torch.cat([y_all,y_z[c]], dim=0)
                x_all = x_all.cuda()
                y_all = y_all.cuda()

                torch.manual_seed(my_seed)

                output_model = model(x_all) # output_base, output_c1, output_c2, ...,
                like = torch.sum(F.softmax(output_model, dim=1)*y_all,1)
                like_z = torch.t(like.view(num_k, batch_size))
                pi_z = probs[operation1_z,operation2_z].cuda()
                hc_numerator = torch.clamp(pi_z*like_z,1e-10,1.0)
                hc_denominator = torch.sum(hc_numerator,dim=1).unsqueeze(1)

                hz_n_org = torch.clamp(hc_numerator / hc_denominator / args.sig ,1e-10,1.0)
                hz_n = F.softmax(-hz_n_org, dim=1)
                hz_n_w = torch.t(hz_n).contiguous().view(num_k*batch_size)
                probs_count_batch = torch.zeros(probs.size())
                for c in range(num_k):
                    for x,y,z in zip(operation1_z[:,c],operation2_z[:,c],hz_n[:,c]): 
                        probs_count_batch[x,y] +=z

                i = i_batch % probs_count.size(0)
                probs_count[i] = probs_count_batch
                probs = torch.sum(probs_count,dim=0)
                probs = probs / torch.sum(probs)

                mag = torch.zeros(batch_size,10).cuda()
                mag1_z = mag1_z.long()
                mag2_z = mag2_z.long()

                for c in range(num_k):
                    idx = torch.arange(c*batch_size,(c+1)*batch_size)
                    onehot_mag1 = torch.eye(10)[mag1_z[:,c]]
                    mag += onehot_mag1.cuda()*hz_n[:,c].view(-1,1)
                    onehot_mag2 = torch.eye(10)[mag2_z[:,c]]
                    mag += onehot_mag2.cuda()*hz_n[:,c].view(-1,1)
                mag_cand += torch.sum(mag,dim=0)

                if args.cutout == 1:
                    if "svhn" in args.dataset:
                        x_all = Cutout_batch(x_all, 20, num_k)
                    else:
                        x_all = Cutout_batch(x_all, x_all.size(2)//2, num_k)

                if args.cutmix == 1:
                    x_all, y_all = Cutmix_batch(x_all, y_all, num_k)

            torch.manual_seed(my_seed)

            output = model(x_all)
            like_output = torch.sum(y_all * nn.functional.log_softmax(output, dim=1), dim=1)
            loss = -torch.mean(num_k*hz_n_w*like_output)

            target = torch.argmax(y_all, dim=1)
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            losses.update(loss.data.item(), output.size(0))
            top1.update(prec1.item(), output.size(0))
            top5.update(prec5.item(), output.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(
                train_acc="{:.4f}".format(top1.avg),
                train_loss="{:.4f}".format(losses.avg),
                )
            )

    scheduler.step()

    return top1.avg, top5.avg, losses.avg, probs, probs_count, mag_cand

def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with tqdm(val_loader, ncols=150, leave=False, disable=args.disable_tqdm) as pbar:
        pbar.set_description("[val:   %d]" % (epoch+1))
        for x, y0, x_list, y_list, operation1_z, operation2_z, mag1_z, mag2_z  in pbar:
            x = x.cuda()
            target = torch.argmax(y0, dim=1).cuda()

            with torch.no_grad():
                output = model(x)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            losses.update(loss.data.item(), output.size(0))
            top1.update(prec1.item(), output.size(0))
            top5.update(prec5.item(), output.size(0))

            pbar.set_postfix(OrderedDict(
                va_acc="{:.4f}".format(top1.avg),
                val_loss="{:.4f}".format(losses.avg),
                )
            )
    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()
