python3 train.py --dataset cifar10 \
      --name cifar10-wrn40x2 \
      --dataroot /home/kkuri/Dropbox/temp/PyTorch/data/ \
      --checkpoint /home/kkuri/results/runs/latent/mixld5/ \
      --model wrn \
      --num_k 6 \
      --epochs 200 \
      --batch-size 128 \
      --lr 0.1 \
      --model wrn40x2 \
      --weight-decay 0.0002 \
      --layers 40 \
      --widen-factor 2 \
      --cutmix_prob 0.5 \
      --cutmix True