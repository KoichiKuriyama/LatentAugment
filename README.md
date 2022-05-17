# LatentAugment

The optimal augmentation policy, which is the latent variable, cannot be directly observed.
- LatentAugment estimates the probability of latent augmentation using the EM algorithm. 
- LatentAugment is simple and computationally efficient. It can estimate using the simple stochastic gradient descent algorithm without an adversarial network.
- LatentAugment has higher test accuracy than previous augmentation methods on the CIFAR-10, CIFAR-100, SVHN, and ImageNet datasets.

## Results
|  Dataset  |          Model          |    Baseline   |       AA      |     AdvAA     |  UBS  | MA  |       LA  (proposed)     |
|:---------:|:-----------------------:|:-------------:|:-------------:|:-------------:|:-----:|:------:|:------------:|
|  CIFAR-10 |     Wide-ResNet-40-2    |     94.70     |     96.30     |       -       |   -   | 96.79 | <b>97.27</b>  |
|           |    Wide-ResNet-28-10    |     96.13     |     97.32     |     98.10     | 97.89 | 97.76 |  <b>98.25</b>  |
|           |  Shake-Shake (26 2x32d) |     96.45     |     97.53     |     97.64     |   -   |    -   |   <b> 97.68</b>   |
|           |  Shake-Shake (26 2x96d) |     97.14     |     98.01     |     98.15     | 98.27 |  98.29 |   <b>98.42</b>   |
|           | Shake-Shake (26 2x112d) |     97.18     |     98.11     |     98.22     |   -   |  98.28 | <b>98.44</b>   |
|           |   PyramidNet+ShakeDrop  |     97.33     |     98.52     |     98.64     | 98.66 |  98.57 |  <b>98.72</b>   |
| CIFAR-100 |     Wide-ResNet-40-2    |     74.00     |     79.30     |       -       |   -   |  80.60 |  <b>80.90</b>   |
|           |    Wide-ResNet-28-10    |     81.20     |     82.91     |     84.51     | 84.54 |   83.79 | <b>84.98</b>   |
|           |  Shake-Shake (26 2x96d) |     82.95     |     85.72     |    85.90     |   -   |  <b> 85.97</b> | 85.88  |
|    SVHN   |    Wide-ResNet-28-10    |     98.50     |     98.93     |       -       |   -   |     -   |    <b>98.96 </b>     |
|  ImageNet |        ResNet-50        | 75.30 / 92.20 | 77.63 / 93.82 | 79.40 / 94.47 |   -   |  79.74 / 94.64 | <b>80.02 / 94.88</b>  |

 AutoAugment (AA) (Cubuk et al., 2018), Adversarial AutoAugment (AdvAA) (Zhang et al., 2019), Uncertainty-Based Sampling (UBS) (Wu et al., 2020), MetaAugment (MA) (Zhou et al., 2020), and proposed LatentAugment (LA). 

## Usage
wrn40x2 model with cifar10 dataset:
```
$ python train.py --dataset cifar10 \
      --name cifar10-wrn40x2 \
      --dataroot /home/user/data/ \
      --checkpoint /home/user/runs/latent/ \
      --num_k 6 \
      --epochs 200 \
      --batch-size 128 \
      --lr 0.1 \
      --weight-decay 0.0002 \
      --model wrn \
      --layers 40 \
      --widen-factor 2 \
      --cutmix_prob 0.5 \
      --cutmix True
```

For other models and datasets, you can find script files in the script folder.
