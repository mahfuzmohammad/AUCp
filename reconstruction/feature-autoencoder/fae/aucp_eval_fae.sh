#!/bin/bash

# python aucp_eval_fae.py --seed 0 --train_dataset rsna --loss_fn ssim --no_train;
# python aucp_eval_fae.py --seed 0 --train_dataset vin --loss_fn ssim --no_train;
# python aucp_eval_fae.py --seed 0 --train_dataset brain --loss_fn ssim --no_train;
# python aucp_eval_fae.py --seed 0 --train_dataset lag --loss_fn ssim --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset isic --loss_fn ssim --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset c16 --loss_fn ssim --no_train;

python aucp_eval_fae.py --seed 0 --train_dataset rsna --loss_fn mse --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset vin --loss_fn mse --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset brain --loss_fn mse --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset lag --loss_fn mse --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset isic --loss_fn mse --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset c16 --loss_fn mse --no_train;

python aucp_eval_fae.py --seed 0 --train_dataset brats --loss_fn ssim --no_train;
python aucp_eval_fae.py --seed 0 --train_dataset brats --loss_fn mse --no_train;
