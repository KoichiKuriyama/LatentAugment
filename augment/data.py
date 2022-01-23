mport logging
import os

import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF
import math

from torchvision.transforms import transforms
from augment.policy import *
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import ConcatDataset
import torchvision.datasets as datasets
import torch.utils.data.distributed

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_IMAGENT_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


_SVHN_MEAN, _SVHN_STD = (0.43090966, 0.4302428, 0.44634357), (0.19652855, 0.19832038, 0.19942076) # from PBA


def get_dataloaders(dataset, batch, dataroot, num_classes, num_workers, args, num_k=4, num_transforms=16, horovod=False):

    if dataset == 'cifar10':
        total_trainset = MyCifarDataset(datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None),
        dataset_name=dataset, train=True, batch=batch, num_classes=10, num_k=num_k, num_transforms=num_transforms)
        testset = MyCifarDataset(datasets.CIFAR10(root=dataroot, train=False, download=True, transform=None),
        dataset_name=dataset, train=False, batch=batch, num_classes=10, num_k=num_k, num_transforms=num_transforms)
    elif dataset == 'cifar100':
        total_trainset = MyCifarDataset(datasets.CIFAR100(root=dataroot, train=True, download=True, transform=None),
        dataset_name=dataset, train=True, batch=batch, num_classes=100, num_k=num_k, num_transforms=num_transforms)
        testset = MyCifarDataset(datasets.CIFAR100(root=dataroot, train=False, download=True, transform=None),
        dataset_name=dataset, train=False, batch=batch, num_classes=100, num_k=num_k, num_transforms=num_transforms)
    elif dataset == 'svhn':
        trainset = datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        extraset = datasets.SVHN(root=dataroot, split='extra', download=True, transform=None)
        total_trainset = ConcatDataset([trainset, extraset])
        total_trainset = MySVHNDataset(total_trainset, dataset_name=dataset, train=True, batch=batch, num_classes=10,
        num_k=num_k, num_transforms=num_transforms)
        testset = MySVHNDataset(datasets.SVHN(root=dataroot, split='test', download=True, transform=None),
        dataset_name=dataset, train=False, batch=batch, num_classes=10, num_k=num_k, num_transforms=num_transforms)
    elif dataset == 'svhn_core':
        trainset = datasets.SVHN(root=dataroot, split='train', download=True, transform=None)
        total_trainset = MySVHNDataset(trainset, dataset_name=dataset, train=True, batch=batch, num_classes=10,
        num_k=num_k, num_transforms=num_transforms)
        testset = MySVHNDataset(datasets.SVHN(root=dataroot, split='test', download=True, transform=None),
        dataset_name=dataset, train=False, batch=batch, num_classes=10, num_k=num_k, num_transforms=num_transforms)
    elif 'imagenet' in dataset:
        total_trainset = MyImagenetDataset(datasets.ImageFolder(os.path.join(dataroot, 'train'),None),
        dataset_name=dataset, train=True, batch=batch, num_classes=1000, num_k=num_k, num_transforms=num_transforms)
        testset = MyImagenetDataset(datasets.ImageFolder(os.path.join(dataroot, 'val'),None),
        dataset_name=dataset, train=False, batch=batch, num_classes=1000, num_k=num_k, num_transforms=num_transforms)

    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False)

    return trainloader, testloader, total_trainset, testset, train_sampler

class MyCifarDataset(Dataset):
    def __init__(self, data_source, dataset_name, train, batch, num_classes, num_k, num_transforms):
        self.mydata = data_source
        self.train = train
        if "cifar" in dataset_name:
            self.data_mean = _CIFAR_MEAN
            self.data_std = _CIFAR_STD
        else:
            raise ValueError('invalid dataset_name=%s' % dataset_name)
        self.num_classes = num_classes
        self.num_k = num_k
        self.num_transforms = num_transforms
        self.latent_policy = LatentPolicy()
        self.batch = batch
        self.mixup_id = 15 # mixup id of LatentPolicy

    def __getitem__(self, index):
        image, target = self.mydata[index]
        y1 = torch.eye(self.num_classes)[target]
        y0 = y1.clone()
        operation1_z = torch.zeros(self.num_k).long()
        operation2_z = torch.zeros(self.num_k).long()
        images = torch.zeros((self.num_k,3,)+tuple(image.size))

        y_list = torch.zeros(self.num_k,self.num_classes)
        mag1_list = torch.zeros(self.num_k).long()
        mag2_list = torch.zeros(self.num_k).long()
        if self.train == True:
            image0 = image.copy()
            for c in range(-1,self.num_k):  # -1: image_base
                if c > -1:
                    image1 = image.copy()
                    image1 = transforms.RandomCrop(32, padding=4)(image1)
                    image1 = transforms.RandomHorizontalFlip()(image1)
                    operation1_z[c] = random.randint(0, self.num_transforms - 1)
                    operation2_z[c] = random.randint(0, self.num_transforms - 1)

                    if operation1_z[c] == self.mixup_id:
                        index2 = random.randint(0, len(self.mydata)-1) # mixup
                        image2, target2 = self.mydata[index2]
                        y2 = torch.eye(self.num_classes)[target2]
                        image2 = transforms.RandomCrop(32, padding=4)(image2)
                        image2 = transforms.RandomHorizontalFlip()(image2)
                    else:
                        image2 = image1
                        y2 = y1
                    if operation2_z[c] == self.mixup_id:
                        index3 = random.randint(0, len(self.mydata)-1) # mixup
                        image3, target3 = self.mydata[index3]
                        y3 = torch.eye(self.num_classes)[target3]
                        image3 = transforms.RandomCrop(32, padding=4)(image3)
                        image3 = transforms.RandomHorizontalFlip()(image3)
                    else:
                        image3 = image1
                        y3 = y1

                    image1, y1, mag1, mag2 = self.latent_policy(image1, image2, image3, y0, y2, y3, operation1_z[c], operation2_z[c])

                    # Transform to tensor
                    image1 = TF.to_tensor(image1)
                    # Normalize
                    image1 = TF.normalize(image1,self.data_mean, self.data_std)
                    images[c,:,:,:]=image1
                    y_list[c] = y1
                    mag1_list[c] = mag1
                    mag2_list[c] = mag2

                else:
                    image_base = image.copy()
                    image_base = transforms.RandomCrop(32, padding=4)(image_base)
                    image_base = transforms.RandomHorizontalFlip()(image_base)
                    # Transform to tensor
                    image_base = TF.to_tensor(image_base)
                    # Normalize
                    image_base = TF.normalize(image_base,self.data_mean, self.data_std)
                    #image_base = image1
                    mag1 = 0
                    mag2 = 0

        else:
            # Transform to tensor
            image = TF.to_tensor(image)

            # Normalize
            image = TF.normalize(image,self.data_mean, self.data_std)

            images = image
            image_base = image

        #y1: one-hot-target
        return image_base, y0, images, y_list, operation1_z, operation2_z, mag1_list, mag2_list

    def __len__(self):
        return len(self.mydata)

class MySVHNDataset(Dataset):
    def __init__(self, data_source, dataset_name, train, batch, num_classes, num_k, num_transforms):
        self.mydata = data_source
        self.train = train
        self.data_mean = _SVHN_MEAN
        self.data_std = _SVHN_STD
        self.num_classes = num_classes
        self.num_k = num_k
        self.num_transforms = num_transforms
        self.latent_policy = LatentPolicy()
        self.batch = batch
        self.mixup_id = 15 # mixup id of LatentPolicy

    def __getitem__(self, index):
        image, target = self.mydata[index]
        y1 = torch.eye(self.num_classes)[target]
        y0 = y1.clone()
        operation1_z = torch.zeros(self.num_k).long()
        operation2_z = torch.zeros(self.num_k).long()
        images = torch.zeros((self.num_k,3,)+tuple(image.size))

        y_list = torch.zeros(self.num_k,self.num_classes)
        mag1_list = torch.zeros(self.num_k).long()
        mag2_list = torch.zeros(self.num_k).long()
        if self.train == True:
            image0 = image.copy()

            for c in range(-1,self.num_k):  # -1: image_base

                if c > -1:
                    image1 = image.copy()
                    image1 = transforms.RandomCrop(32, padding=4)(image1)
                    #image1 = transforms.RandomHorizontalFlip()(image1)
                    operation1_z[c] = random.randint(0, self.num_transforms - 1)
                    operation2_z[c] = random.randint(0, self.num_transforms - 1)

                    if operation1_z[c] == self.mixup_id:
                        index2 = random.randint(0, len(self.mydata)-1) # mixup
                        image2, target2 = self.mydata[index2]
                        image2 = transforms.RandomCrop(32, padding=4)(image2)
                        #image2 = transforms.RandomHorizontalFlip()(image2)
                        y2 = torch.eye(self.num_classes)[target2]
                    else:
                        image2 = image1
                        y2 = y0

                    if operation2_z[c] == self.mixup_id:
                        index3 = random.randint(0, len(self.mydata)-1) # mixup
                        image3, target3 = self.mydata[index3]
                        image3 = transforms.RandomCrop(32, padding=4)(image3)
                        #image3 = transforms.RandomHorizontalFlip()(image3)
                        y3 = torch.eye(self.num_classes)[target3]
                    else:
                        image3 = image0
                        y3 = y0

                    # LatentPolicy: y0, y2 and y3 are one-hot-targets
                    image1, y1, mag1, mag2 = self.latent_policy(image1, image2, image3, y0, y2, y3, operation1_z[c], operation2_z[c])

                    # Random Crop
                    #image1 = transforms.RandomCrop(32, padding=4)(image1)

                    # Transform to tensor
                    image1 = TF.to_tensor(image1)
                    # Normalize
                    image1 = TF.normalize(image1,self.data_mean, self.data_std)
                    images[c,:,:,:]=image1
                    y_list[c] = y1
                    mag1_list[c] = mag1
                    mag2_list[c] = mag2

                else:
                    image_base = image.copy()
                    #image_base = transforms.RandomCrop(32, padding=4)(image_base)
                    #image_base = transforms.RandomHorizontalFlip()(image_base)

                    # Transform to tensor
                    image_base = TF.to_tensor(image0)
                    # Normalize
                    image_base = TF.normalize(image_base,self.data_mean, self.data_std)
                    #image_base = image1
                    mag1 = 0
                    mag2 = 0

        else:
            # Transform to tensor
            image = TF.to_tensor(image)

            # Normalize
            image = TF.normalize(image,self.data_mean, self.data_std)

            images = image
            image_base = image

        #y1: one-hot-target
        return image_base, y0, images, y_list, operation1_z, operation2_z, mag1_list, mag2_list

    def __len__(self):
        return len(self.mydata)

class MyImagenetDataset(Dataset):
    def __init__(self, data_source, dataset_name, train, batch, num_classes, num_k, num_transforms):
        self.mydata = data_source
        self.train = train
        self.data_mean = _IMAGENT_MEAN
        self.data_std = _IMAGENET_STD
        self.num_classes = num_classes
        self.num_k = num_k
        self.num_transforms = num_transforms
        self.latent_policy = LatentPolicy()
        self.batch = batch
        self.mixup_id = 15 # mixup id of LatentPolicy

    def __getitem__(self, index):
        image, target = self.mydata[index]
        y1 = torch.eye(self.num_classes)[target]
        y0 = y1.clone()
        operation1_z = torch.zeros(self.num_k).long()
        operation2_z = torch.zeros(self.num_k).long()
        images = torch.zeros((self.num_k,3,224,224))

        y_list = torch.zeros(self.num_k,self.num_classes)
        mag1_list = torch.zeros(self.num_k).long()
        mag2_list = torch.zeros(self.num_k).long()
        if self.train == True:
            image0 = image.copy()

            for c in range(-1,self.num_k):  # -1: image_base
                if c > -1:
                    operation1_z[c] = random.randint(0, self.num_transforms - 1)
                    operation2_z[c] = random.randint(0, self.num_transforms - 1)

                    image1 = transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC)(image0)
                    image1 = transforms.RandomHorizontalFlip()(image1)
                    image1 = transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    )(image1)

                    if operation1_z[c] == self.mixup_id:
                        index2 = random.randint(0, len(self.mydata)-1) # Mixup
                        image2, target2 = self.mydata[index2]
                        y2 = torch.eye(self.num_classes)[target2]
                        image2 = transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC)(image2)
                        image2 = transforms.RandomHorizontalFlip()(image2)
                        image2 = transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                        )(image2)
                    else:
                        image2 = image1
                        y2 = y0

                    if operation2_z[c] == self.mixup_id:
                        index3 = random.randint(0, len(self.mydata)-1) # Mixup
                        image3, target3 = self.mydata[index3]
                        y3 = torch.eye(self.num_classes)[target3]
                        image3 = transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC)(image3)
                        image3 = transforms.RandomHorizontalFlip()(image3)
                        image3 = transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                        )(image3)
                    else:
                        image3 = image1
                        y3 = y0

                    image1, y1, mag1, mag2 = self.latent_policy(image1, image2, image3, y0, y2, y3, operation1_z[c], operation2_z[c])

                    image1 = transforms.ToTensor()(image1)
                    image1 = Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec'])(image1)
                    image1 = transforms.Normalize(mean=self.data_mean, std=self.data_std)(image1)

                    images[c,:,:,:]=image1
                    y_list[c] = y1
                    mag1_list[c] = mag1
                    mag2_list[c] = mag2

                else:
                    image_base = image.copy()
                    image_base = transforms.Resize(256, interpolation=Image.BICUBIC)(image_base)
                    image_base = transforms.CenterCrop(224)(image_base)
                    # Transform to tensor
                    image_base = transforms.ToTensor()(image_base)
                    image_base = Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec'])(image_base)
                    # Normalize
                    image_base = transforms.Normalize(mean=self.data_mean, std=self.data_std)(image_base)
                    #image_base = image1
                    mag1 = 0
                    mag2 = 0

        else:
            image = transforms.Resize(256, interpolation=Image.BICUBIC)(image)
            image = transforms.CenterCrop(224)(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=self.data_mean, std=self.data_std)(image)

            images = image
            image_base = image

        return image_base, y0, images, y_list, operation1_z, operation2_z, mag1_list, mag2_list

    def __len__(self):
        return len(self.mydata)

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
