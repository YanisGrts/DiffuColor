import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from unet import UNet
import numpy as np
from skimage import color
import os 
from cocoloader import COCOColorizationDataset
import numpy as np
from skimage import color
# --- Config ---
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    entity="DeepColo",
    project="Unet",
    config={"batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS},
    name="Unet",
)

# --- Data ---
# Replace these with your actual dataset splits
train_dataset = COCOColorizationDataset("../ds/coco/train2017")
val_dataset   = COCOColorizationDataset("../ds/coco/val2017")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- Model ---
model     = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

wandb.watch(model, log="gradients", log_freq=100)  # optional: tracks gradient norms

# --- Training loop ---
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for L, ab in train_loader:
        L, ab = L.to(DEVICE), ab.to(DEVICE)

        pred = model(L)
        loss = criterion(pred, ab)

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
            val_loss += criterion(model(L), ab).item()
    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | train {avg_train:.4f} | val {avg_val:.4f}")
    wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch + 1})
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch+1}.pt"))
        model.eval()
        with torch.no_grad():
            L_sample, ab_sample = next(iter(val_loader))
            L_sample = L_sample[:4].to(DEVICE)
            pred_ab  = model(L_sample)

            images = []
            for i in range(4):
                L_np  = (L_sample[i, 0].cpu().numpy() + 1.0) * 50.0 
                ab_np = (pred_ab[i].cpu().numpy() * 110.0)
                lab   = np.stack([L_np, ab_np[0], ab_np[1]], axis=-1)
                rgb   = (color.lab2rgb(lab) * 255).astype(np.uint8)
                images.append(wandb.Image(rgb, caption=f"sample {i}"))

            wandb.log({"colorizations": images})
