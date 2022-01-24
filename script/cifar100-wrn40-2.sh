python train.py --dataset cifar100 \
      --name cifar100-wrn40x2 \
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