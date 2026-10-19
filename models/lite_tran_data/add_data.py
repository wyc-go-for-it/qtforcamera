import torch
import time
from torch.utils.data import DataLoader, ConcatDataset, random_split

device = torch.device("cuda")

# 1. 原有数据
duts_ds = DUTSDatasetSize("/content/DUTS-TR/DUTS-TR-Image",
                          "/content/DUTS-TR/DUTS-TR-Mask", size=320)
food_ds = DUTSDatasetSize("/content/foodseg/images",
                          "/content/foodseg/masks", size=320)

# 2. 新增数据
new_ds = DUTSDatasetSize("/content/new_data/images",
                         "/content/new_data/masks", size=320)

print("DUTS:", len(duts_ds), "Food:", len(food_ds), "New:", len(new_ds))

# 3. 新数据按比例加权（如果新数据量小）
# 方案 A：不加权，直接混
full = ConcatDataset([duts_ds, food_ds, new_ds])

# 方案 B：新数据重复 N 倍（新数据 < 3000 张时用）
# repeat = max(1, 10000 // len(new_ds))
# full = ConcatDataset([duts_ds, food_ds] + [new_ds] * repeat)

print("total:", len(full))

n_val = 500
n_train = len(full) - n_val
train_ds, val_ds = random_split(full, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          num_workers=2, persistent_workers=True,
                          pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False,
                          num_workers=2, persistent_workers=True,
                          pin_memory=True)

# 4. 从 lite_best.pth 初始化（关键：不是从头）
net = LiteSaliencyNet().to(device)
net.load_state_dict(torch.load("/content/lite_best.pth", map_location=device))
print("loaded lite_best.pth (IoU 0.9309)")

# 5. 正常学习率（不是微调的 1e-4）
EPOCHS = 20
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = BCEDiceLoss().to(device)

best_iou = 0.9309   # 从当前 best 开始，只有超过才保存
CKPT = "/content/lite_ckpt2.pth"
BEST = "/content/lite_best_v2.pth"

import os
start_epoch = 0
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=device)
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_iou = ckpt["best_iou"]
    print(f"resume from epoch {start_epoch}")

def compute_iou(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter + 1e-6
    return (inter / union).item()

for epoch in range(start_epoch, EPOCHS):
    net.train()
    t0 = time.time()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = net(imgs)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    net.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            pred = net(imgs)
            val_loss += criterion(pred, masks).item()
            val_iou  += compute_iou(pred, masks)

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    val_iou    /= len(val_loader)

    print(f"[{epoch+1:02d}/{EPOCHS}] train_loss={train_loss:.4f} "
          f"val_loss={val_loss:.4f} IoU={val_iou:.4f} time={time.time()-t0:.1f}s")

    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_iou": best_iou,
    }, CKPT)

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(net.state_dict(), BEST)
        print(f"  ✅ saved best IoU={best_iou:.4f}")

print("done, best IoU:", best_iou)