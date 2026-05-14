# This code was produced with the assistance of Claude (Anthropic). The authors reviewed and validated the content.
import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from skimage import color
from scipy import linalg
import warnings


# Colorfulness Score
def colorfulness_score(rgb_img: np.ndarray) -> float:
    """
    Compute the colorfulness metric
    """
    rgb = rgb_img.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    rg_mean, rg_std = rg.mean(), rg.std()
    yb_mean, yb_std = yb.mean(), yb.std()

    std_rgyb = math.sqrt(rg_std**2 + yb_std**2)
    mean_rgyb = math.sqrt(rg_mean**2 + yb_mean**2)

    return std_rgyb + 0.3 * mean_rgyb


def get_inception_model(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def preprocess_for_inception(rgb_imgs):
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return torch.stack([tfm(Image.fromarray(img)) for img in rgb_imgs])


@torch.no_grad()
def compute_activations(imgs_np, inception, device, batch_size=32):
    """Return Inception pool3 features: (N, 2048)."""
    all_acts = []
    for i in range(0, len(imgs_np), batch_size):
        batch = preprocess_for_inception(imgs_np[i:i+batch_size]).to(device)
        acts = inception(batch)
        all_acts.append(acts.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def compute_fid(acts_real, acts_gen):
    """Frechet Inception Distance. Lower is better."""
    mu_r, sigma_r = acts_real.mean(0), np.cov(acts_real, rowvar=False)
    mu_g, sigma_g = acts_gen.mean(0),  np.cov(acts_gen,  rowvar=False)

    diff = mu_r - mu_g

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


# Lab -> RGB conversion
def lab_to_rgb(L_norm, ab_norm):
    L = (L_norm + 1.0) * 50.0
    ab = ab_norm * 128.0
    lab = np.stack([L, ab[0], ab[1]], axis=-1)
    rgb = (color.lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)
    return rgb


def save_images(save_dir, img_idx, grayscale_np, real_rgb, gen_rgbs):
    folder = os.path.join(save_dir, f"img_{img_idx:04d}")
    os.makedirs(folder, exist_ok=True)

    L_uint8 = ((grayscale_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    grey_rgb = np.stack([L_uint8, L_uint8, L_uint8], axis=-1)
    Image.fromarray(grey_rgb).save(os.path.join(folder, "grayscale.png"))
    Image.fromarray(real_rgb).save(os.path.join(folder, "real.png"))

    for v, rgb in enumerate(gen_rgbs):
        name = f"var_{v+1}.png" if len(gen_rgbs) > 1 else "unet.png"
        Image.fromarray(rgb).save(os.path.join(folder, name))


def sample_ddpm(model, L_batch, alpha_bars, betas, alphas, T, guidance_scale, device):
    """Run full reverse diffusion for a batch. Returns ab (B,2,H,W) """
    from noisescheduler import p_sample

    B, _, H, W = L_batch.shape
    ab_gen = torch.randn(B, 2, H, W, device=device)

    for step in reversed(range(T)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)
        ab_gen = p_sample(model, ab_gen, L_batch, t_batch,
                          alpha_bars, betas, alphas,
                          guidance_scale=guidance_scale)
    return ab_gen.cpu()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",            choices=["ddpm", "unet"], required=True)
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--data_dir",        type=str, default="../ds/coco/val2017")
    p.add_argument("--n_images",        type=int, default=500,
                   help="Number of val images to evaluate on")
    p.add_argument("--n_variations",    type=int, default=4,
                   help="DDPM only: colorization samples per image")
    p.add_argument("--guidance_scale",  type=float, default=1.5,
                   help="DDPM only: CFG guidance scale")
    p.add_argument("--batch_size",      type=int, default=8,
                   help="Images processed per GPU batch (lower if OOM)")
    p.add_argument("--inception_batch", type=int, default=32,
                   help="Batch size for Inception feature extraction")
    p.add_argument("--save_dir",        type=str, default=None,
                   help="If set, save grayscale / real / generated images here. "
                        "Each input image gets its own subfolder (img_XXXX/).")
    p.add_argument("--skip_fid",        action="store_true",
                   help="Skip FID computation (fast sanity-check mode)")
    return p.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(
        f"[eval] Device: {DEVICE} | Mode: {args.mode} | Images: {args.n_images}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(
            f"[eval] Images will be saved to: {os.path.abspath(args.save_dir)}")

    from cocoloader import COCOColorizationDataset
    dataset = COCOColorizationDataset(args.data_dir)
    indices = list(range(min(args.n_images, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4,
                        pin_memory=DEVICE.type == "cuda")

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt

    if args.mode == "ddpm":
        from ddpm import UNetDDPM
        from noisescheduler import alpha_bars, betas, alphas, T

        model = UNetDDPM().to(DEVICE)
        model.load_state_dict(state)
        model.eval()

        alpha_bars_d = alpha_bars.to(DEVICE)
        betas_d = betas.to(DEVICE)
        alphas_d = alphas.to(DEVICE)

    else:
        import sys
        sys.path.insert(0, "../unet")
        from unet import UNet

        model = UNet().to(DEVICE)
        model.load_state_dict(state)
        model.eval()

    print("[eval] Collecting real images ...")
    real_rgb_all = []
    grayscale_all = []

    for L_batch, ab_batch in loader:
        for i in range(L_batch.shape[0]):
            L_np = L_batch[i, 0].numpy()
            ab_np = ab_batch[i].numpy()
            grayscale_all.append(L_np)
            real_rgb_all.append(lab_to_rgb(L_np, ab_np))

    print("[eval] Generating colorizations ...")
    gen_rgb_all = []   # flat list of every generated image
    cf_scores = []   # one value per (image, variation)
    gen_per_image = [[] for _ in range(len(real_rgb_all))]

    global_img_idx = 0

    with torch.no_grad():
        for L_batch, _ in loader:
            L_batch = L_batch.to(DEVICE)
            B = L_batch.shape[0]

            if args.mode == "ddpm":
                batch_variations = []
                # Run N variations
                for _ in range(args.n_variations):
                    ab_gen = sample_ddpm(model, L_batch,
                                         alpha_bars_d, betas_d, alphas_d,
                                         T, args.guidance_scale, DEVICE)
                    var_rgbs = []
                    for i in range(B):
                        rgb = lab_to_rgb(L_batch[i, 0].cpu().numpy(),
                                         ab_gen[i].numpy())
                        var_rgbs.append(rgb)
                        gen_rgb_all.append(rgb)
                        cf_scores.append(colorfulness_score(rgb))
                    batch_variations.append(var_rgbs)

                for i in range(B):
                    img_idx = global_img_idx + i
                    for v in range(args.n_variations):
                        gen_per_image[img_idx].append(batch_variations[v][i])

            else:
                pred_ab = model(L_batch).cpu()
                for i in range(B):
                    rgb = lab_to_rgb(L_batch[i, 0].cpu().numpy(),
                                     pred_ab[i].numpy())
                    gen_rgb_all.append(rgb)
                    cf_scores.append(colorfulness_score(rgb))
                    gen_per_image[global_img_idx + i].append(rgb)

            global_img_idx += B
            print(
                f"  ... {global_img_idx}/{len(real_rgb_all)} images done", end="\r")

    print()

    if args.save_dir:
        print(f"[eval] Saving images ...")
        for idx in range(len(real_rgb_all)):
            save_images(
                save_dir=args.save_dir,
                img_idx=idx,
                grayscale_np=grayscale_all[idx],
                real_rgb=real_rgb_all[idx],
                gen_rgbs=gen_per_image[idx],
            )
        print(
            f"[eval] Saved {len(real_rgb_all)} sets to {os.path.abspath(args.save_dir)}")
        print(f"       Layout: img_XXXX/ -> grayscale.png | real.png | var_1.png ...")

    # CF
    cf_arr = np.array(cf_scores)

    print("\n-- Colorfulness Score (CF) -----------------------------------------")
    if args.mode == "ddpm":
        cf_per_image = cf_arr.reshape(-1, args.n_variations)
        cf_mean_per_image = cf_per_image.mean(axis=1)
        cf_std_per_image = cf_per_image.std(axis=1)
        print(
            f"   Mean CF (avg over variations): {cf_mean_per_image.mean():.3f}")
        print(
            f"   Std  CF (avg over variations): {cf_mean_per_image.std():.3f}")
        print(
            f"   Mean intra-image variation:    {cf_std_per_image.mean():.3f}")
        print(f"   (higher intra-image variation -> more diverse colorizations)")
    else:
        print(f"   Mean CF: {cf_arr.mean():.3f}")
        print(f"   Std  CF: {cf_arr.std():.3f}")

    # FID
    if not args.skip_fid:
        print("\n[eval] Computing Inception features for FID ...")
        inception = get_inception_model(DEVICE)

        print("  -> real images ...")
        acts_real = compute_activations(
            real_rgb_all, inception, DEVICE, args.inception_batch)
        print("  -> generated images ...")
        acts_gen = compute_activations(
            gen_rgb_all,  inception, DEVICE, args.inception_batch)

        fid = compute_fid(acts_real, acts_gen)
        print(f"\n-- FID -------------------------------------------------------------")
        print(f"   FID score: {fid:.2f}  (lower is better)")
        print(f"   Real pool:      {len(real_rgb_all)} images")
        print(f"   Generated pool: {len(gen_rgb_all)} images")
    else:
        print("\n[eval] FID skipped (--skip_fid).")

    fid_str = f"{fid:.2f}" if fid is not None else "skipped"
    print(f"""
+------------------------------------------------------+
|  EVALUATION SUMMARY - {args.mode.upper():<6}                        |
+------------------------------------------------------+
|  Checkpoint : {os.path.basename(args.checkpoint):<38}|
|  Images     : {args.n_images:<38}|""")

    if args.mode == "ddpm":
        print(f"|  Variations : {args.n_variations:<38}|")
        print(f"|  CFG scale  : {args.guidance_scale:<38}|")

    print(f"""|  CF (mean)  : {cf_arr.mean():<38.3f}|
|  FID        : {fid_str:<38}|
+------------------------------------------------------+""")

    if args.save_dir:
        print(f"\nImages saved -> {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
