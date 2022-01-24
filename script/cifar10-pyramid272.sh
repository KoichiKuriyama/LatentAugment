python train.py --dataset cifar10 \
      --name cifar10-pyramid272 \
      --dataroot /home/user/data/ \
      --checkpoint /home/user/runs/latent/ \
      --num_k 6 \
      --epochs 1800 \
      --batch-size 64 \
      --lr 0.05 \
      --weight-decay 5.0e-5 \
      --model pyramid \
      --depth = 272 \
      --alpha = 200 \
      --bottleneck = True \
      --cutmix_prob 0.5 \
      --cutmix True
