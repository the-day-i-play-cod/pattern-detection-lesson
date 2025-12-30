"""读取视网膜病变数据、构建 PyTorch Dataset 并训练 EfficientNetV2 的示例代码。（改版）"""

import argparse
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import models, transforms
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class SampleRecord:
    path: Path
    retinopathy_grade: int
    macular_edema_risk: int

TARGET_FIELD_ATTRS = {
    "retinopathy": "retinopathy_grade",
    "macular_edema": "macular_edema_risk",
}


class RetinaDataset(Dataset):
    """把 images 和 Excel 表格对齐，提供给训练管线的 Dataset。"""

    def __init__(
        self,
        image_root: Path,
        annotation_root: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.transform = transform or self.default_transforms()
        self.samples = self._collect_samples()
        self._check_samples()

    def default_transforms(self) -> transforms.Compose:
        return build_transforms(384)

    def _build_image_index(self) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = {}
        for img_path in self.image_root.rglob("*.*"):
            if not img_path.is_file():
                continue
            key = img_path.name.lower()
            index.setdefault(key, []).append(img_path)
        return index

    def _find_column(self, columns: Sequence[str], target: str) -> Optional[str]:
        normalized = target.strip().lower()
        for col in columns:
            if str(col).strip().lower() == normalized:
                return col
        return None


    def _collect_samples(self) -> List[SampleRecord]:
        index = self._build_image_index()
        samples: List[SampleRecord] = []
        for excel in sorted(self.annotation_root.glob("Annotation*.xls*")):
            base_hint = self._guess_base_name(excel.stem)
            try:
                df = pd.read_excel(excel, dtype=str)
            except ValueError:
                df = pd.read_excel(excel, dtype=str, engine="xlrd")
            image_col = self._find_column(df.columns, "Image name")
            grade_col = self._find_column(df.columns, "Retinopathy grade")
            risk_col = self._find_column(df.columns, "Risk of macular edema")
            if not image_col or not grade_col:
                continue
            for _, row in df.iterrows():
                image_key = str(row.get(image_col, "")).strip()
                if not image_key:
                    continue
                candidates = index.get(image_key.lower())
                if not candidates:
                    continue
                img_path = self._select_candidate(candidates, base_hint)
                grade = self._parse_int(row.get(grade_col))
                risk = self._parse_int(row.get(risk_col))
                if grade is None:
                    continue
                samples.append(SampleRecord(img_path, grade, risk or 0))
        return samples

    def _guess_base_name(self, text: str) -> Optional[str]:
        match = re.search(r"Base(\d{2})", text, re.IGNORECASE)
        return f"Base{match.group(1)}" if match else None

    def _select_candidate(self, candidates: Sequence[Path], base_hint: Optional[str]) -> Path:
        if not base_hint or len(candidates) == 1:
            return candidates[0]
        for path in candidates:
            if base_hint in path.parts:
                return path
        return candidates[0]

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _check_samples(self) -> None:
        if not self.samples:
            raise RuntimeError("没有找到任何图像/标注匹配项，检查路径和 Excel 内容")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        record = self.samples[idx]
        try:
            image = Image.open(record.path).convert("RGB")
        except Exception as e:
            print(f"警告: 跳过损坏图像 {record.path}: {e}")
            # 递归获取下一个有效图像，防止无限循环
            next_idx = (idx + 1) % len(self.samples)
            if next_idx == idx:  # 仅一个样本时
                raise RuntimeError("数据集中只有损坏的图像")
            return self.__getitem__(next_idx)
        image = self.transform(image)
        return (
            image,
            {
                "retinopathy": torch.tensor(record.retinopathy_grade, dtype=torch.long),
                "macular_edema": torch.tensor(record.macular_edema_risk, dtype=torch.long),
            },
        )


def build_model(num_classes: int) -> nn.Module:
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT
    except AttributeError:
        weights = None
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _to_logits(outputs: Union[torch.Tensor, object]) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    return logits if isinstance(logits, torch.Tensor) else outputs  # type: ignore[return-value]


def build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_weights_and_sampler(
    train_subset: torch.utils.data.Subset,
    dataset: RetinaDataset,
    target_attr: str,
    num_classes: int,
) -> Tuple[torch.Tensor, WeightedRandomSampler]:
    targets = [getattr(dataset.samples[i], target_attr) for i in train_subset.indices]
    counts = Counter(targets)
    class_weights = torch.tensor(
        [1.0 / max(1, counts.get(c, 0)) for c in range(num_classes)], dtype=torch.float
    )
    sample_weights = [1.0 / max(1, counts[t]) for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return class_weights, sampler


def plot_history(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    if plt is None:
        print("matplotlib 未安装，跳过绘图。")
        return
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    acc = [h["val_acc"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, loss, "b-o", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, "g-o", label="val_acc")
    ax2.set_ylabel("val_acc", color="g")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"训练曲线已保存: {out_path}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_key: str,
) -> float:
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="训练", leave=False):
        images = images.to(device)
        labels = targets[target_key].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        logits = _to_logits(outputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_key: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="验证", leave=False):
            images = images.to(device)
            labels = targets[target_key].to(device)
            outputs = model(images)
            logits = _to_logits(outputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def evaluate_with_counts(
    model: nn.Module, loader: DataLoader, device: torch.device, target_key: str
) -> Dict[str, Dict[int, int]]:
    model.eval()
    counter: Dict[str, Dict[int, int]] = {
        "total": {},
        "correct": {},
    }
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="评估", leave=False):
            images = images.to(device)
            labels = targets[target_key].to(device)
            outputs = model(images)
            logits = _to_logits(outputs)
            preds = logits.argmax(dim=1)
            for label, pred in zip(labels.tolist(), preds.tolist()):
                counter["total"][label] = counter["total"].get(label, 0) + 1
                if label == pred:
                    counter["correct"][label] = counter["correct"].get(label, 0) + 1
    return counter


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "MESSDIOR",
        help="包含 images/ 和 annotation of images/ 的根路径",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-samples", action="store_true", help="只打印前十个样本并退出")
    parser.add_argument(
        "--target-field",
        choices=list(TARGET_FIELD_ATTRS.keys()),
        default="retinopathy",
        help="训练/验证时使用的目标字段(macular_edema 或 retinopathy)",
    )
    args = parser.parse_args()

    generator = _set_seed(args.seed)
    image_root = args.root / "images"
    annotation_root = args.root / "annotation of images"
    dataset = RetinaDataset(image_root, annotation_root)
    target_attr = TARGET_FIELD_ATTRS[args.target_field]
    target_values = [getattr(record, target_attr) for record in dataset.samples]
    if not target_values:
        raise RuntimeError("样本列表为空，无法确定类别数")
    num_classes = max(target_values) + 1

    val_count = int(len(dataset) * args.val_ratio)
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)

    # 构建加权采样与加权损失，缓解类别不平衡
    class_weights, sampler = build_weights_and_sampler(train_ds, dataset, target_attr, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.debug_samples:
        for record in dataset.samples[:10]:
            print(record)
        dist = {grade: sum(1 for r in dataset.samples if r.retinopathy_grade == grade) for grade in range(max(r.retinopathy_grade for r in dataset.samples) + 1)}
        print(f"样本总数={len(dataset)}, 类别分布={dist}")
        return

    history: List[Dict[str, float]] = []
    print(f"使用目标字段: {args.target_field}")

    # 分阶段分辨率训练：224 → 320 → 384
    stage_sizes = [224, 320, 384]
    stages = len(stage_sizes)
    base = args.epochs // stages
    rem = args.epochs % stages
    stage_epochs = [base + (1 if i < rem else 0) for i in range(stages)]
    current_epoch = 0

    for stage_idx, (size, num_stage_epochs) in enumerate(zip(stage_sizes, stage_epochs), 1):
        if num_stage_epochs == 0:
            continue
        dataset.transform = build_transforms(size)
        print(f"阶段{stage_idx}: 输入分辨率 {size}×{size}, 轮数={num_stage_epochs}")
        for _ in range(num_stage_epochs):
            current_epoch += 1
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                target_key=args.target_field,
            )
            acc = evaluate(model, val_loader, device, target_key=args.target_field)
            history.append(
                {
                    "epoch": current_epoch,
                    "loss": loss,
                    "val_acc": acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            print(f"Epoch {current_epoch}/{args.epochs}: loss={loss:.4f} val_acc={acc:.4f} (size={size})")

    print("训练记录:")
    for record in history:
        print(f" epoch={int(record['epoch'])} loss={record['loss']:.4f} val_acc={record['val_acc']:.4f} lr={record['lr']:.6f}")

    plot_history(history, Path("training_curve.png"))

    final_counts = evaluate_with_counts(model, val_loader, device, target_key=args.target_field)
    print("验证集各类正确率:")
    for label, total in sorted(final_counts["total"].items()):
        correct = final_counts["correct"].get(label, 0)
        pct = correct / total if total else 0
        print(f"  class {label}: {correct}/{total} ({pct:.2%})")


if __name__ == "__main__":
    main()