import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from ddpm import UNetDDPM
import numpy as np
from skimage import color
import os 
from cocoloader import COCOColorizationDataset
from noisescheduler import alpha_bars, q_sample, T, p_sample, betas, alphas

# --- Config ---
BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

run = wandb.init(
    entity="DeepColo",
    project="Unet",
    config={"batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS},
    name="DDPM",
)

# --- Data ---
train_dataset = COCOColorizationDataset("../ds/coco/train2017")
val_dataset   = COCOColorizationDataset("../ds/coco/val2017")

CHECKPOINT_DIR = f"checkpoints/{run.id}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- Model ---
model     = UNetDDPM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
alpha_bars = alpha_bars.to(DEVICE)
betas = betas.to(DEVICE)
alphas = alphas.to(DEVICE)
wandb.watch(model, log="gradients", log_freq=100)  # optional: tracks gradient norms


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for L, ab in train_loader:
        L, ab = L.to(DEVICE), ab.to(DEVICE)

        t     = torch.randint(0, T, (L.shape[0],), device=DEVICE)
        noise = torch.randn_like(ab)
        ab_noisy   = q_sample(ab, t, noise, alpha_bars)

        pred_noise = model(ab_noisy, L, t)
        loss       = nn.functional.mse_loss(pred_noise, noise)  # MSE here

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            t     = torch.randint(0, T, (L.shape[0],), device=DEVICE)
            noise = torch.randn_like(ab)
            ab_noisy   = q_sample(ab, t, noise, alpha_bars)
            pred_noise = model(ab_noisy, L, t)
            val_loss  += nn.functional.mse_loss(pred_noise, noise).item()
    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | train {avg_train:.4f} | val {avg_val:.4f}")
    wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch + 1})
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch+1}.pt"))
        model.eval()
        with torch.no_grad():
            L_sample, _ = next(iter(val_loader))
            L_sample = L_sample[:2].to(DEVICE)
            L_repeated = L_sample.repeat_interleave(4, dim=0)

            ab_gen = torch.randn(8, 2, 128, 128, device=DEVICE)
            for step in reversed(range(T)):
                t_batch = torch.full((8,), step, device=DEVICE, dtype=torch.long)
                ab_gen = p_sample(model, ab_gen, L_repeated, t_batch, alpha_bars, betas, alphas)

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
