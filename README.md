# LatentAugment

The optimal augmentation policy, which is the latent variable, cannot be directly observed.
- LatentAugment estimates the probability of latent augmentation. 
- LatentAugment is simple and computationally efficient.

## Results
|  Dataset  |          Model          |    Baseline   |       AA      |     AdvAA     |  UBS  |       LA  (proposed)     |
|:---------:|:-----------------------:|:-------------:|:-------------:|:-------------:|:-----:|:-------------:|
|  CIFAR-10 |     Wide-ResNet-40-2    |     94.70     |     96.30     |       -       |   -   |  <b>97.27</b>  |
|           |    Wide-ResNet-28-10    |     96.13     |     97.32     |     98.10     | 97.89 |   <b>98.25</b>  |
|           |  Shake-Shake (26 2x32d) |     96.45     |     97.53     |     97.64     |   -   |   <b> 97.68</b>   |
|           |  Shake-Shake (26 2x96d) |     97.14     |     98.01     |     98.15     | 98.27 |   <b>98.42</b>   |
|           | Shake-Shake (26 2x112d) |     97.18     |     98.11     |     98.22     |   -   |   <b>98.44</b>   |
|           |   PyramidNet+ShakeDrop  |     97.33     |     98.52     |     98.64     | 98.66 |   <b>98.72</b>   |
| CIFAR-100 |     Wide-ResNet-40-2    |     74.00     |     79.30     |       -       |   -   |   <b>80.90</b>   |
|           |    Wide-ResNet-28-10    |     81.20     |     82.91     |     84.51     | 84.54 |   <b>84.98</b>   |
|           |  Shake-Shake (26 2x96d) |     82.95     |     85.72     |     <b>85.90</b>     |   -   |   85.88  |
|    SVHN   |    Wide-ResNet-28-10    |     98.50     |     98.93     |       -       |   -   |     <b>98.99 </b>     |
|  ImageNet |        ResNet-50        | 75.30 / 92.20 | 77.63 / 93.82 | 79.40 / 94.47 |   -   | <b>79.40 / 94.47</b>  |

