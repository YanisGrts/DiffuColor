import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from ddpm import UNetDDPM
import numpy as np
from skimage import color
import os
import argparse
from cocoloader import COCOColorizationDataset
from noisescheduler import alpha_bars, q_sample, T, p_sample, betas, alphas


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM colorization model")
    parser.add_argument("--epochs",         type=int,   default=30,   help="Number of epochs to train")
    parser.add_argument("--batch_size",     type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",             type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint",     type=str,   default=None, help="Path to a checkpoint .pt file to resume training from")
    # Classifier-free guidance
    parser.add_argument("--drop_prob",      type=float, default=0.15, help="Probability of dropping L condition during training (CFG)")
    parser.add_argument("--guidance_scale", type=float, default=2.0,  help="CFG guidance scale at sampling time (1.0 = no guidance)")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    BATCH_SIZE     = args.batch_size
    LR             = args.lr
    EPOCHS         = args.epochs
    DROP_PROB      = args.drop_prob
    GUIDANCE_SCALE = args.guidance_scale
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Wandb ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity="DeepColo",
        project="Unet",
        config={
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
            "epochs":         EPOCHS,
            "drop_prob":      DROP_PROB,
            "guidance_scale": GUIDANCE_SCALE,
        },
        name="DDPM-CFG",
        resume="allow",
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    train_dataset = COCOColorizationDataset("../ds/coco/train2017")
    val_dataset   = COCOColorizationDataset("../ds/coco/val2017")

    CHECKPOINT_DIR = f"checkpoints/{run.id}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = UNetDDPM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Cosine LR schedule — decays smoothly to lr/10 over the run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 10)

    alpha_bars_d = alpha_bars.to(DEVICE)
    betas_d      = betas.to(DEVICE)
    alphas_d     = alphas.to(DEVICE)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_epoch = 0
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        print(f"[resume] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"[resume] Resuming from epoch {start_epoch + 1}")
        else:
            # Legacy plain state-dict
            model.load_state_dict(ckpt)
            basename = os.path.basename(args.checkpoint)
            try:
                start_epoch = int("".join(filter(str.isdigit, basename.split("epoch")[-1].split(".")[0])))
                print(f"[resume] Inferred start epoch {start_epoch} from filename")
            except (ValueError, IndexError):
                print("[resume] Could not infer epoch from filename; starting epoch counter at 0")

        # Fast-forward the LR scheduler to match where we are
        for _ in range(start_epoch):
            scheduler.step()

    wandb.watch(model, log="gradients", log_freq=100)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        train_loss = 0.0

        for L, ab in train_loader:
            L, ab = L.to(DEVICE), ab.to(DEVICE)

            t        = torch.randint(0, T, (L.shape[0],), device=DEVICE)
            noise    = torch.randn_like(ab)
            ab_noisy = q_sample(ab, t, noise, alpha_bars_d)

            # CFG: randomly drop the L condition for a fraction of the batch
            drop_condition = torch.rand(L.shape[0], device=DEVICE) < DROP_PROB

            pred_noise = model(ab_noisy, L, t, drop_condition=drop_condition)
            loss       = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(DEVICE), ab.to(DEVICE)
                t        = torch.randint(0, T, (L.shape[0],), device=DEVICE)
                noise    = torch.randn_like(ab)
                ab_noisy = q_sample(ab, t, noise, alpha_bars_d)
                # Validation always uses the full condition (no dropping)
                pred_noise = model(ab_noisy, L, t)
                val_loss  += nn.functional.mse_loss(pred_noise, noise).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch + 1:3d} | train {avg_train:.4f} | val {avg_val:.4f} | lr {current_lr:.2e}")
        wandb.log({
            "train_loss": avg_train,
            "val_loss":   avg_val,
            "lr":         current_lr,
            "epoch":      epoch + 1,
        })

        # ── Checkpoint every 2 epochs ───────────────────────────────────────────
        if (epoch + 1) % 2 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch + 1}.pt")
            torch.save(
                {
                    "epoch":           epoch + 1,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] Saved {ckpt_path}")

            # ── Sample colorizations with CFG ───────────────────────────────────
            with torch.no_grad():
                L_sample, _ = next(iter(val_loader))
                L_sample    = L_sample[:2].to(DEVICE)
                L_repeated  = L_sample.repeat_interleave(4, dim=0)

                ab_gen = torch.randn(8, 2, 128, 128, device=DEVICE)
                for step in reversed(range(T)):
                    t_batch = torch.full((8,), step, device=DEVICE, dtype=torch.long)
                    ab_gen  = p_sample(
                        model, ab_gen, L_repeated, t_batch,
                        alpha_bars_d, betas_d, alphas_d,
                        guidance_scale=GUIDANCE_SCALE,
                    )

                images = []
                for i in range(8):
                    L_np  = (L_repeated[i, 0].cpu().numpy() + 1.0) * 50.0
                    ab_np = ab_gen[i].cpu().numpy() * 128.0
                    lab   = np.stack([L_np, ab_np[0], ab_np[1]], axis=-1)
                    rgb   = (color.lab2rgb(lab) * 255).astype(np.uint8)

                    img_num = (i // 4) + 1
                    var_num = (i % 4) + 1
                    images.append(wandb.Image(rgb, caption=f"Image {img_num} - Variation {var_num}"))
                wandb.log({"colorizations": images})

    wandb.finish()


if __name__ == "__main__":
    main()