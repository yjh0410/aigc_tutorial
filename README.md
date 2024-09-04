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

### Train VqVAE (not complete ...)

- Taking the `CelebA` as the example, you could refer to the following command to train `VqVAE`.
```Shell
python train_vqvae.py --cuda --max_epoch 100 --eval_epoch 5 --model vqvae --batch_size 128 --dataset celebA --root path/to/CelebA
```

After training, you could refer to the following command to evaluate `VqVAE`.
- Sample on MNIST
```Shell
python sample_vqvae.py --cuda --model vqvae --dataset celebA --root path/to/CelebA
```

You could also refer to the following command to sample with `VqVAE` and `VqVAESampler`.
```Shell
python sample_vqvae.py --cuda --model vqvae --dataset celebA --root path/to/CelebA --sample
```

You could download the pretrained `VqVAE` on `CelebA` dataset from the following link:

`VqVAE` on CelebA: [checkpoint](https://github.com/yjh0410/aigc_tutorial/releases/download/aigc_checkpoints/vqvae_pretrained_on_celebA.pth)
