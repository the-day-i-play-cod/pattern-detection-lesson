import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置参数
images_dir = 'images'  # 图片根目录
batch_size = 32
num_epochs = 100  # 增大训练轮次
num_classes = 4  # 根据实际类别数修改
val_ratio = 0.15
test_ratio = 0.15

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
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

# 1. 自定义数据集
class EyeDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.samples = []
        self.transform = transform
        for base_folder in os.listdir(images_dir):
            base_path = os.path.join(images_dir, base_folder)
            if not os.path.isdir(base_path):
                continue
            for file in os.listdir(base_path):
                if file.endswith('.xls') or file.endswith('.xlsx'):
                    xls_path = os.path.join(base_path, file)
                    df = pd.read_excel(xls_path)
                    img_names = df.iloc[:, 0].values
                    labels = df.iloc[:, 2].values
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = EyeDataset(images_dir, transform)
all_labels = [label for _, label in dataset.samples]
indices = list(range(len(dataset)))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 强制使用GPU
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU is not available! 请在有NVIDIA显卡的环境下运行。')
device = torch.device('cuda')

for fold, (train_idx, testval_idx) in enumerate(kfold.split(indices, all_labels)):
    print(f"Fold {fold+1}/5")
    testval_labels = [all_labels[i] for i in testval_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(val_ratio+test_ratio), random_state=fold)
    val_idx, test_idx = next(sss_val.split(testval_idx, testval_labels))
    val_idx = [testval_idx[i] for i in val_idx]
    test_idx = [testval_idx[i] for i in test_idx]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False)
    }

    # 使用MobileNetV3 Large
    model = models.mobilenet_v3_large(pretrained=True)
    # 替换分类头
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses, val_losses = [] , []
    train_accs, val_accs = [] , []
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for images, labels in tqdm(dataloaders['train'], desc='Train', leave=False):
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
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloaders['val'], desc='Val', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_samples += images.size(0)
        val_epoch_loss = val_loss / val_samples
        val_epoch_acc = val_corrects / val_samples
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        lr_scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f"mobilenetv3_eye_model_best_fold{fold+1}.pth")
            print(f"模型已保存为 mobilenetv3_eye_model_best_fold{fold+1}.pth (best)")
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), f"mobilenetv3_eye_model_fold{fold+1}.pth")
    print(f"模型已保存为 mobilenetv3_eye_model_fold{fold+1}.pth (last)")

    # 绘制损失率变化曲线
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve Fold {fold+1} (MobileNetV3)')
    plt.legend()
    plt.savefig(f'mobilenetv3_loss_curve_fold{fold+1}.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve Fold {fold+1} (MobileNetV3)')
    plt.legend()
    plt.savefig(f'mobilenetv3_acc_curve_fold{fold+1}.png')
    plt.close()

    # 测试集评估，加载最佳模型
    model.load_state_dict(torch.load(f"mobilenetv3_eye_model_best_fold{fold+1}.pth", map_location=device))
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloaders['test'], desc='Test', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels).item()
            test_samples += images.size(0)
    test_epoch_loss = test_loss / test_samples
    test_epoch_acc = test_corrects / test_samples
    print(f"Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}")
