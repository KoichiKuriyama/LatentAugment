python train.py --dataset cifar100 \
      --name cifar100-shake26_2x96d \
      --dataroot /home/user/data/ \
      --checkpoint /home/user/runs/latent/ \
      --num_k 6 \
      --epochs 1800 \
      --batch-size 128 \
      --lr 0.01 \
      --weight-decay 0.0025 \
      --model shakeshake \
      --layers 26 \
      --widen-factor 96 \
      --cutmix_prob 0.5 \
      --cutmix True
