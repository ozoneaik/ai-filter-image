import os, json, time, random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
EPOCHS = 10
NUM_WORKERS = 4
LR = 3e-4
IMG_SIZE = 224
DATA_DIR = "dataset"  # ชี้ไปยังโฟลเดอร์ดาต้า

# Augment & Normalize
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
class_names = train_ds.classes
num_classes = len(class_names)
print("คลาส:", class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

# โมเดลจาก timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc, best_path = 0.0, "best_model.pt"

def evaluate():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total, loss_sum/total

if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc, val_loss = evaluate()
        print(f"Epoch {epoch}: ค่า acc={val_acc:.4f}, ค่า loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names
            }, best_path)
            print(f"✅ บันทึก model ใน {best_path} (acc={best_acc:.4f})")
