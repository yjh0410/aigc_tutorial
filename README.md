# aigc_tutorial
Basic tutorial for getting started with AIGC


## Train AIGC
### Train VAE

- Taking the `mnist` as the example, you could refer to the following command to train `VAE`.
```Shell
python train_vae.py --cuda --max_epoch 20 --eval_epoch 5 --model vae --batch_size 128 --dataset mnist
```

After training, you could refer to the following command to sample with `VAE`.
- Sample on MNIST
```Shell
python sample_vae.py --cuda --model vae --dataset mnist --weight path/to/checkpoint
```

### Train VQ-VAE

- Taking the `CelebA` as the example, you could refer to the following command to train `VQ-VAE`.
```Shell
python train_vqvae.py --cuda --max_epoch 100 --eval_epoch 5 --model vqvae --batch_size 128 --img_size 64 --dataset celebA --root path/to/CelebA
```

After training, you could refer to the following command to evaluate `VQ-VAE`.
- Sample on MNIST
```Shell
python sample_vqvae.py --cuda --model vqvae --img_size 64 --dataset celebA --root path/to/CelebA
```

You could also refer to the following command to sample with `VQ-VAE` and `VqVAESampler`.
```Shell
python sample_vqvae.py --cuda --model vqvae --img_size 64 --dataset celebA --root path/to/CelebA --sample
```

You could download the pretrained `VQ-VAE` on `CelebA` dataset from the following link:

- `VQ-VAE` on CelebA with image size of 64: [vqvae_celebA_size_64](https://github.com/yjh0410/aigc_tutorial/releases/download/aigc_checkpoints/vqvae_on_celebA_size_64.pth)

- `VQ-VAE` on CelebA with image size of 128: [vqvae_celebA_size_128](https://github.com/yjh0410/aigc_tutorial/releases/download/aigc_checkpoints/vqvae_on_celebA_size_128.pth)

- GPTSampler for `VQ-VAE` on CelebA with image size of 64: [vqvae_sampler_gpt_small_celebA_size_64](https://github.com/yjh0410/aigc_tutorial/releases/download/aigc_checkpoints/vqvae_sampler_gpt_small_celebA_size_64.pth)
### Train VQ-GAN

- Taking the `CelebA` as the example, you could refer to the following command to train `VQ-GAN`.
```Shell
python vqgan_train.py --cuda --max_epoch 100 --eval_epoch 5 --batch_size 16 --lr 0.000025 --img_size 64 --dataset celebA --root path/to/CelebA
```

- `VQ-GAN` on CelebA with image size of 128: [vqgan_celebA_size_128](https://github.com/yjh0410/aigc_tutorial/releases/download/aigc_checkpoints/vqgan_celebA_size_128_psnr_30.4_ssim_0.9048.pth)

