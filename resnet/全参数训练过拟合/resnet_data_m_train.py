import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedShuffleSplit

# 配置参数
images_dir = 'images'  # 图片根目录
batch_size = 32
num_epochs = 50
num_classes = 4  # 根据实际类别数修改

# 1. 自定义数据集
class EyeDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.samples = []
        self.transform = transform
        # 遍历所有Base**子目录
        for base_folder in os.listdir(images_dir):
            base_path = os.path.join(images_dir, base_folder)
            if not os.path.isdir(base_path):
                continue
            # 查找xls文件
            for file in os.listdir(base_path):
                if file.endswith('.xls') or file.endswith('.xlsx'):
                    xls_path = os.path.join(base_path, file)
                    df = pd.read_excel(xls_path)
                    img_names = df.iloc[:, 0].values  # 第一列为图片名
                    labels = df.iloc[:, 2].values     # 第三列为Retinopathy grade
                    for img_name, label in zip(img_names, labels):
                        img_path = os.path.join(base_path, img_name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, int(label)))
                        else:
                            print(f"Warning: {img_path} not found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. 数据增强与加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自动划分训练集和验证集（分层采样）
dataset = EyeDataset(images_dir, transform)
all_labels = [label for _, label in dataset.samples]
indices = list(range(len(dataset)))
val_ratio = 0.15  # 验证集比例
sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
train_idx, val_idx = next(sss.split(indices, all_labels))
from torch.utils.data import Subset
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 3. 定义模型（迁移学习，仅微调最后几层）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换最后全连接层
model.fc.requires_grad = True  # 只训练fc层
model = model.to(device)

# 4. 损失函数与优化器（只优化fc层参数）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 记录训练和验证损失/准确率
train_losses = []
train_accs = []
val_losses = []
val_accs = []
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# 5. 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += images.size(0)
        # 打印每个batch的信息
        print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f} Acc: {(torch.sum(preds == labels).item() / images.size(0)):.4f}")
    epoch_loss = running_loss / len(train_set)
    epoch_acc = running_corrects / total_samples
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels).item()
            val_samples += images.size(0)
    val_epoch_loss = val_loss / len(val_set)
    val_epoch_acc = val_corrects / val_samples
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Summary: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    # 早停判断（用验证损失）
    early_stopping(val_epoch_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("训练完成！")

# 保存损失曲线
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('resnet_loss_curve.png')
plt.close()

# 保存准确率曲线
plt.figure()
plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('resnet_acc_curve.png')
plt.close()

torch.save(model.state_dict(), "resnet_eye_model.pth")
print("模型已保存为 resnet_eye_model.pth")