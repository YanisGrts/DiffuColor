Here is a `README.md` file tailored to your project.

---

# DiffuColor: Image Colorization

DiffuColor is a deep learning project focused on image colorization. It converts grayscale images (L channel in LAB color space) into fully colored images by predicting the missing color channels (a and b).

This repository implements two distinct approaches to solve the colorization problem:

1. **Direct Regression (UNet)**: A standard UNet model trained with L1 loss to directly predict the ab channels.
2. **Diffusion Model (DDPM)**: A Denoising Diffusion Probabilistic Model conditioned on the L channel to generate the ab channels over a series of denoising steps.

## Project Structure

* `ds/coco/`: Contains dataset preprocessing scripts.
* `unet/`: Contains the dataset loader, standard UNet architecture, and training script for the direct regression approach.
* `diffusion/`: Contains the noise scheduler, DDPM UNet architecture, dataset loader, and training script for the diffusion-based approach.

## Prerequisites

Before running the scripts, ensure you have the required dependencies installed. You will need:

* Python 3.x
* PyTorch & Torchvision
* `numpy`, `Pillow`, `scikit-image` (`skimage`), `matplotlib`
* `tqdm` (for dataset processing)
* `wandb` (Weights & Biases, for tracking experiments)

You will also need to create a Weights & Biases account and log in using `wandb login` in your terminal, as both training scripts automatically log metrics and sample colorizations to wandb.

## Step 1: Dataset Preparation

This project uses the COCO 2017 dataset.

1. Download the COCO `train2017` and `val2017` image folders.
2. Place them inside the `ds/coco/` directory.
3. The dataset contains some images that are naturally black and white, which can confuse the model. Navigate to the dataset directory and run the preprocessing script to resize the images and filter out B&W photos:
```bash
cd ds/coco
python modifyDS.py

```

## Step 2: Training

You can choose to train either the standard UNet or the Diffusion model.

### Option A: Train the Standard UNet

The standard UNet takes the L channel and directly outputs the a and b channels using an L1 Loss criterion.

```bash
cd unet
python train.py

```

This script will save checkpoints in `unet/checkpoints/{wandb_run_id}/` every 5 epochs and push visual samples.


### Option B: Train the Diffusion Model (DDPM)

The diffusion model gradually denoises random noise into valid a and b channels, conditioned on the L channel.

```bash
cd diffusion
python train.py
```

This script will save checkpoints in `diffusion/checkpoints/{wandb_run_id}/` every 2 epochs and push visual samples.

