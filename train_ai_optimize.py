import os, json, time, random
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่า seed สำหรับผลลัพธ์ที่สม่ำเสมอ
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
EPOCHS = 20  # เพิ่ม epochs
NUM_WORKERS = 4
LR = 1e-4  # ลด learning rate เล็กน้อย
WEIGHT_DECAY = 1e-4  # เพิ่ม weight decay
IMG_SIZE = 224
DATA_DIR = "dataset"
PATIENCE = 5  # สำหรับ early stopping

print(f"🚀 กำลังเริ่มการเทรน AI สำหรับจำแนกภาพ")
print(f"📱 Device: {device}")
print(f"🎯 Classes: GREETING, NSFW, OTHER")

# Data Augmentation ที่แข็งแกร่งขึ้น
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # ขยายขนาดก่อน crop
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),  # Random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # เพิ่ม vertical flip
    transforms.RandomRotation(15),  # เพิ่มมุมหมุน
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # เพิ่ม affine transform
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),  # เพิ่ม random erasing
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# โหลดข้อมูล
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
class_names = train_ds.classes
num_classes = len(class_names)

print(f"📊 จำนวนคลาส: {num_classes}")
print(f"📁 คลาส: {class_names}")

# คำนวณจำนวนตัวอย่างในแต่ละคลาส
class_counts = Counter([train_ds.targets[i] for i in range(len(train_ds))])
print(f"📈 จำนวนข้อมูลแต่ละคลาส:")
for i, class_name in enumerate(class_names):
    print(f"   {class_name}: {class_counts[i]} ภาพ")

# สร้าง WeightedRandomSampler สำหรับ class imbalance
class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float)
sample_weights = [class_weights[train_ds.targets[i]] for i in range(len(train_ds))]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoader ด้วย sampler
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH, 
    sampler=sampler,  # ใช้ sampler แทน shuffle
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True  # เพิ่มประสิทธิภาพ
)

val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

# โมเดลที่ทันสมัยกว่า
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
model = model.to(device)

print(f"🧠 โมเดล: efficientnet_b0")
print(f"📦 จำนวน parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss function ด้วย class weights
class_weights_tensor = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer และ Scheduler ที่ดีขึ้น
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Scheduler แบบ warm restart
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=5, 
    T_mult=2,
    eta_min=1e-6
)

best_acc = 0.0
best_path = "best_model_enhanced.pt"
patience_counter = 0
train_losses = []
val_accuracies = []

def calculate_accuracy_per_class(model, dataloader):
    """คำนวณ accuracy แต่ละคลาส"""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            
            for i in range(y.size(0)):
                label = y[i].item()
                class_correct[label] += (pred[i] == y[i]).item()
                class_total[label] += 1
    
    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    
    return class_acc

def evaluate():
    """ฟังก์ชันประเมินผล"""
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
    print(f"\n🎯 เริ่มการเทรน...")
    print("="*60)
    
    for epoch in range(1, EPOCHS+1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping เพื่อป้องกัน gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Validation phase
        val_acc, val_loss = evaluate()
        avg_train_loss = epoch_loss / num_batches
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d}/{EPOCHS}")
        print(f"  📈 Train Loss: {avg_train_loss:.4f}")
        print(f"  🎯 Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  📚 Learning Rate: {current_lr:.2e}")
        
        # บันทึกโมเดลที่ดีที่สุด
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # คำนวณ accuracy แต่ละคลาส
            class_acc = calculate_accuracy_per_class(model, val_loader)
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_acc": val_acc,
                "class_accuracies": class_acc,
                "train_losses": train_losses,
                "val_accuracies": val_accuracies
            }, best_path)
            
            print(f"  ✅ บันทึกโมเดลใหม่! Accuracy: {best_acc:.4f}")
            print(f"     📊 Accuracy แต่ละคลาส:")
            for i, (class_name, acc) in enumerate(zip(class_names, class_acc)):
                print(f"       {class_name}: {acc:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
        
        print("-" * 60)
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping! ไม่มีการปรับปรุงใน {PATIENCE} epochs")
            break
    
    print(f"\n🎉 เทรนเสร็จสิ้น!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    print(f"💾 โมเดลถูกบันทึกที่: {best_path}")
    
    # โหลดโมเดลที่ดีที่สุดและแสดงผลสุดท้าย
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    final_class_acc = calculate_accuracy_per_class(model, val_loader)
    print(f"\n📊 Final Class Accuracies:")
    for class_name, acc in zip(class_names, final_class_acc):
        print(f"  {class_name}: {acc:.4f}")
    
    print(f"\n💡 Tips สำหรับการใช้งาน:")
    print(f"  - ใช้โมเดลนี้สำหรับจำแนกภาพ: GREETING, NSFW, OTHER")
    print(f"  - ขนาดภาพที่แนะนำ: {IMG_SIZE}x{IMG_SIZE} pixels")
    print(f"  - โมเดลใช้ efficientnet_b0 architecture")