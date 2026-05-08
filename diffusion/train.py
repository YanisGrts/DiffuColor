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
    parser.add_argument("--epochs",      type=int,   default=30,   help="Number of epochs to train")
    parser.add_argument("--batch_size",  type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    BATCH_SIZE = args.batch_size
    LR         = args.lr
    EPOCHS     = args.epochs
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Wandb ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity="DeepColo",
        project="Unet",
        config={"batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS},
        name="DDPM",
        resume="allow",   # allows attaching to a previous run when using --checkpoint
        save_code =True,
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

        # Support both plain state-dicts and richer checkpoint dicts
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)   # epoch that was just *completed*
            print(f"[resume] Resuming from epoch {start_epoch + 1}")
        else:
            # Legacy: plain state-dict saved with torch.save(model.state_dict(), …)
            model.load_state_dict(ckpt)
            # Try to infer start_epoch from the filename, e.g. unet_epoch10.pt → 10
            basename = os.path.basename(args.checkpoint)
            try:
                start_epoch = int("".join(filter(str.isdigit, basename.split("epoch")[-1].split(".")[0])))
                print(f"[resume] Inferred start epoch {start_epoch} from filename")
            except (ValueError, IndexError):
                print("[resume] Could not infer epoch from filename; starting epoch counter at 0")

    wandb.watch(model, log="gradients", log_freq=100)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        train_loss = 0.0
        for L, ab in train_loader:
            L, ab = L.to(DEVICE), ab.to(DEVICE)

            t          = torch.randint(0, T, (L.shape[0],), device=DEVICE)
            noise      = torch.randn_like(ab)
            ab_noisy   = q_sample(ab, t, noise, alpha_bars_d)

            pred_noise = model(ab_noisy, L, t)
            loss       = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(DEVICE), ab.to(DEVICE)
                t          = torch.randint(0, T, (L.shape[0],), device=DEVICE)
                noise      = torch.randn_like(ab)
                ab_noisy   = q_sample(ab, t, noise, alpha_bars_d)
                pred_noise = model(ab_noisy, L, t)
                val_loss  += nn.functional.mse_loss(pred_noise, noise).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch + 1:3d} | train {avg_train:.4f} | val {avg_val:.4f}")
        wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch + 1})

        # ── Checkpoint every 5 epochs ───────────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch + 1}.pt")
            # Save a rich checkpoint so future --checkpoint resumes carry optimizer state too
            torch.save(
                {
                    "epoch":           epoch + 1,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] Saved {ckpt_path}")

            # ── Sample colorizations ────────────────────────────────────────────
            with torch.no_grad():
                L_sample, _ = next(iter(val_loader))
                L_sample    = L_sample[:2].to(DEVICE)
                L_repeated  = L_sample.repeat_interleave(4, dim=0)

                ab_gen = torch.randn(8, 2, 128, 128, device=DEVICE)
                for step in reversed(range(T)):
                    t_batch = torch.full((8,), step, device=DEVICE, dtype=torch.long)
                    ab_gen  = p_sample(model, ab_gen, L_repeated, t_batch, alpha_bars_d, betas_d, alphas_d)

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