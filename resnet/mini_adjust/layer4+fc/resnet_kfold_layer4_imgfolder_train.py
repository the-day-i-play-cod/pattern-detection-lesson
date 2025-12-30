import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 配置参数
root_dir = 'eye_prj_data'  # 数据根目录
num_folds = 5
batch_size = 32
num_epochs = 50
num_classes = 4  # 类别数

# 图像增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fold_results = []
with open('resnet_kfold_log.txt', 'w') as flog:
    for fold in range(1, num_folds + 1):
        print(f'========== Fold {fold}/{num_folds} =========')
        flog.write(f'Fold {fold}/{num_folds}\n')
        train_dir = os.path.join(root_dir, f'fold{fold}', 'train')
        val_dir = os.path.join(root_dir, f'fold{fold}', 'val')
        test_dir = os.path.join(root_dir, f'fold{fold}', 'test')
        train_set = datasets.ImageFolder(train_dir, transform=transform)
        val_set = datasets.ImageFolder(val_dir, transform=transform)
        test_set = datasets.ImageFolder(test_dir, transform=transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet18(pretrained=True)
        # 解冻layer4和fc层
        for name, param in model.named_parameters():
            if name.startswith('layer4') or name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

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

        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        best_val_acc = 0
        best_model_state = None
        for epoch in range(num_epochs):
            print(f'--- Epoch {epoch+1}/{num_epochs} ---')
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
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
                batch_acc = torch.sum(preds == labels).item() / images.size(0)
                flog.write(f"Fold {fold} Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.4f}\n")
                print(f"[Fold {fold}][Epoch {epoch+1}/{num_epochs}][Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.4f}")
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
            flog.write(f"Fold {fold} Epoch [{epoch+1}/{num_epochs}] Summary: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n")
            print(f"[Fold {fold}] Epoch [{epoch+1}/{num_epochs}] Summary: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_model_state = model.state_dict()
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                flog.write(f"Fold {fold} Early stopping at epoch {epoch+1}\n")
                break

        # 测试集评估
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        test_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels).item()
                test_samples += images.size(0)
        test_epoch_loss = test_loss / len(test_set)
        test_epoch_acc = test_corrects / test_samples
        flog.write(f"Fold {fold} Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}\n\n")
        fold_results.append({'fold': fold, 'val_acc': best_val_acc, 'test_acc': test_epoch_acc})

        # 保存每个fold的曲线
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve Fold {fold}')
        plt.legend()
        plt.savefig(f'resnet_loss_curve_fold{fold}.png')
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
        plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curve Fold {fold}')
        plt.legend()
        plt.savefig(f'resnet_acc_curve_fold{fold}.png')
        plt.close()

# 保存交叉验证分析结果
with open('resnet_kfold_summary.txt', 'w') as fsum:
    for res in fold_results:
        fsum.write(f"Fold {res['fold']}: Val Acc={res['val_acc']:.4f}, Test Acc={res['test_acc']:.4f}\n")
    avg_val = sum([r['val_acc'] for r in fold_results]) / num_folds
    avg_test = sum([r['test_acc'] for r in fold_results]) / num_folds
    fsum.write(f"Average Val Acc={avg_val:.4f}, Average Test Acc={avg_test:.4f}\n")
print('交叉验证完成，结果已保存。')
