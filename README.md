# DiffuColor: Image Colorization

DiffuColor is a deep learning project focused on image colorization. It converts grayscale images (L channel in LAB color space) into fully colored images by predicting the missing color channels (a and b).

This repository implements two distinct approaches to solve the colorization problem:

1. **Direct Regression (UNet)**: A standard UNet model trained with L1 loss to directly predict the ab channels.
2. **Diffusion Model (DDPM)**: A Denoising Diffusion Probabilistic Model conditioned on the L channel to generate the ab channels over a series of denoising steps.

## Project Structure

* `ds/coco/`: Contains dataset preprocessing scripts.
* `unet/`: Contains the dataset loader, standard UNet architecture, and training script for the direct regression approach.
* `diffusion/`: Contains the noise scheduler, DDPM UNet architecture, dataset loader, and training script for the diffusion-based approach.
