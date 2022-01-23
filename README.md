# LatentAugment

The optimal augmentation policy, which is the latent variable, cannot be directly observed.
- LatentAugment estimates the probability of latent augmentation. 
- LatentAugment is simple and computationally efficient.

## Results
|  Dataset  |          Model          |    Baseline   |       AA      |     AdvAA     |  UBS  |       LA      |
|:---------:|:-----------------------:|:-------------:|:-------------:|:-------------:|:-----:|:-------------:|
|  CIFAR-10 |     Wide-ResNet-40-2    |     94.70     |     96.30     |       -       |   -   |   97.27±0.09  |
|           |    Wide-ResNet-28-10    |     96.13     |     97.32     |     98.10     | 97.89 |   98.25±0.08  |
|           |  Shake-Shake (26 2x32d) |     96.45     |     97.53     |     97.64     |   -   |   97.68±0.03  |
|           |  Shake-Shake (26 2x96d) |     97.14     |     98.01     |     98.15     | 98.27 |   98.42±0.02  |
|           | Shake-Shake (26 2x112d) |     97.18     |     98.11     |     98.22     |   -   |   98.44±0.02  |
|           |   PyramidNet+ShakeDrop  |     97.33     |     98.52     |     98.64     | 98.66 |   98.72±0.02  |
| CIFAR-100 |     Wide-ResNet-40-2    |     74.00     |     79.30     |       -       |   -   |   80.90±0.15  |
|           |    Wide-ResNet-28-10    |     81.20     |     82.91     |     84.51     | 84.54 |   84.98±0.12  |
|           |  Shake-Shake (26 2x96d) |     82.95     |     85.72     |     85.90     |   -   |   85.88±0.10  |
|    SVHN   |    Wide-ResNet-28-10    |     98.50     |     98.93     |       -       |   -   |     98.99     |
|  ImageNet |        ResNet-50        | 75.30 / 92.20 | 77.63 / 93.82 | 79.40 / 94.47 |   -   | 78.29 / 93.98 |

