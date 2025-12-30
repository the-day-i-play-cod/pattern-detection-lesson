"""读取视网膜病变数据、构建 PyTorch Dataset 并训练 GoogLeNet 的示例代码。"""

import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREPROCESS_MEAN = torch.tensor([0.485, 0.456, 0.406])
PREPROCESS_STD = torch.tensor([0.229, 0.224, 0.225])



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
        samples: Optional[List[SampleRecord]] = None,
    ) -> None:
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.transform = transform or self.default_transforms()
        self.samples = samples if samples is not None else self._collect_samples()
        self._check_samples()

    def default_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
        image = Image.open(record.path).convert("RGB")
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
        weights = models.GoogLeNet_Weights.DEFAULT
    except AttributeError:
        weights = None
    model = models.googlenet(weights=weights, aux_logits=weights is not None)
    input_dim = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(input_dim, num_classes)
    return model


def _to_logits(outputs: torch.Tensor | object) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    return logits if isinstance(logits, torch.Tensor) else outputs  # type: ignore[return-value]


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


def _format_shape(shape: Sequence[int]) -> str:
    if len(shape) >= 4:
        c, h, w = shape[1], shape[2], shape[3]
        return f"{c}*{h}*{w}"
    if len(shape) >= 2:
        return "*".join(str(dim) for dim in shape[1:])
    return "*".join(str(dim) for dim in shape)


def summarize_model_layers(model: nn.Module) -> List[str]:
    params_iter = list(model.parameters())
    dummy_device = params_iter[0].device if params_iter else torch.device("cpu")
    input_tensor = torch.zeros(1, 3, 224, 224, device=dummy_device)
    input_shapes: Dict[str, Sequence[int]] = {}
    hooks = []

    def register_hook(name: str, module: nn.Module) -> None:
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], *_: object) -> None:
            if inputs:
                input_shapes[name] = tuple(inputs[0].shape)

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_children():
        register_hook(name, module)

    training = model.training
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    if training:
        model.train()
    for hook in hooks:
        hook.remove()

    lines: List[str] = []
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        shape = input_shapes.get(name)
        shape_info = _format_shape(shape) if shape else "unknown"
        lines.append(
            f"{name:20} {module.__class__.__name__:30} input={shape_info}"
        )
    return lines


def save_training_plot(history: Sequence[Dict[str, float]], dest_dir: Path) -> Path:
    if not history:
        raise ValueError("No history to plot")
    dest_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    losses = [record["loss"] for record in history]
    train_accs = [record.get("train_acc", 0.0) for record in history]
    val_accs = [record["val_acc"] for record in history]

    fig, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(epochs, losses, label="train loss", color="tab:blue", linewidth=2)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, train_accs, label="train acc", color="C1", linestyle="--", linewidth=2)
    ax_acc.plot(epochs, val_accs, label="test acc", color="C2", linestyle="-.", linewidth=2)
    ax_acc.set_ylabel("accuracy")

    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(handles_loss + handles_acc, labels_loss + labels_acc, loc="best")

    fig.tight_layout()
    plot_path = dest_dir / "training_progress.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = PREPROCESS_MEAN.view(-1, 1, 1)
    std = PREPROCESS_STD.view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def dump_preprocessed_samples(
    dataset: RetinaDataset,
    dest: Path,
    count: int,
    seed: int,
    target_key: str,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    total = len(dataset)
    if total == 0:
        return
    indices = rng.sample(range(total), min(count, total))
    to_pil = transforms.ToPILImage()
    for idx in indices:
        image_tensor, targets = dataset[idx]
        preview = denormalize(image_tensor.cpu())
        label = targets[target_key].item()
        filename = dest / f"sample_{idx}_{target_key}_{label}.png"
        to_pil(preview).save(filename)


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def stratified_split_indices(
    samples: Sequence[SampleRecord],
    target_attr: str,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, record in enumerate(samples):
        label = getattr(record, target_attr)
        label_to_indices[label].append(idx)
    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in label_to_indices.values():
        rng.shuffle(indices)
        if not indices:
            continue
        val_count = int(round(len(indices) * val_ratio))
        if val_ratio > 0 and val_count == 0:
            val_count = 1
        val_count = min(len(indices), val_count)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "MESSDIOR",
        help="包含 images/ 和 annotation of images/ 的根路径",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-samples", action="store_true", help="只打印前十个样本并退出")
    parser.add_argument(
        "--dump-samples",
        action="store_true",
        help="把预处理后的图像复制到磁盘",
    )
    parser.add_argument(
        "--dump-count",
        type=int,
        default=20,
        help="dump_samples 时随机保存的图像数量",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="保存预处理图像的目录（默认 workspace/preprocessed_samples）",
    )
    parser.add_argument(
        "--target-field",
        choices=list(TARGET_FIELD_ATTRS.keys()),
        default="retinopathy",
        help="训练/验证时使用的目标字段(macular_edema 或 retinopathy)",
    )
    parser.add_argument(
        "--print-model-info",
        action="store_true",
        help="只打印模型结构/参数后退出",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    image_root = args.root / "images"
    annotation_root = args.root / "annotation of images"
    base_dataset = RetinaDataset(image_root, annotation_root)
    samples = base_dataset.samples
    target_attr = TARGET_FIELD_ATTRS[args.target_field]
    target_values = [getattr(record, target_attr) for record in samples]
    if not target_values:
        raise RuntimeError("样本列表为空，无法确定类别数")
    num_classes = max(target_values) + 1

    # Apply a strong augmentation pipeline for training to improve robustness.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            # transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Use a deterministic resize+normalize pipeline for validation/dumping so metrics stay consistent.
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dump_samples:
        dump_dir = args.dump_dir or Path(__file__).resolve().parent / "preprocessed_samples"
        dump_dataset = RetinaDataset(image_root, annotation_root, transform=eval_transform, samples=samples)
        dump_preprocessed_samples(
            dump_dataset,
            dump_dir,
            count=args.dump_count,
            seed=args.seed,
            target_key=args.target_field,
        )
    train_indices, val_indices = stratified_split_indices(
        samples,
        target_attr,
        args.val_ratio,
        args.seed,
    )
    train_dataset = RetinaDataset(image_root, annotation_root, transform=train_transform, samples=samples)
    val_dataset = RetinaDataset(image_root, annotation_root, transform=eval_transform, samples=samples)
    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    if args.print_model_info:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for line in summarize_model_layers(model):
            print(line)
        print(f"Total params: {total_params:,}; trainable: {trainable_params:,}")
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.debug_samples:
        for record in samples[:10]:
            print(record)
        dist = {grade: sum(1 for r in samples if r.retinopathy_grade == grade) for grade in range(max(r.retinopathy_grade for r in samples) + 1)}
        print(f"样本总数={len(samples)}, 类别分布={dist}")
        return

    history: List[Dict[str, float]] = []
    print(f"使用目标字段: {args.target_field}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            target_key=args.target_field,
        )
        train_acc = evaluate(model, train_loader, device, target_key=args.target_field)
        val_acc = evaluate(model, val_loader, device, target_key=args.target_field)
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: loss={loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    print("训练记录:")
    for record in history:
        print(
            f" epoch={int(record['epoch'])} loss={record['loss']:.4f} train_acc={record['train_acc']:.4f} val_acc={record['val_acc']:.4f} lr={record['lr']:.6f}"
        )

    plot_dir = Path.cwd() / "training_plots"
    plot_path = save_training_plot(history, plot_dir)
    print(f"训练曲线已保存到: {plot_path}")

    final_counts = evaluate_with_counts(model, val_loader, device, target_key=args.target_field)
    print("验证集各类正确率:")
    for label, total in sorted(final_counts["total"].items()):
        correct = final_counts["correct"].get(label, 0)
        pct = correct / total if total else 0
        print(f"  class {label}: {correct}/{total} ({pct:.2%})")

    if history:
        best_record = max(history, key=lambda rec: rec["val_acc"])
        best_val = best_record["val_acc"]
        best_mis = 1.0 - best_val
        print(f"最佳验证准确率={best_val:.2%}，最低错分率={best_mis:.2%}")


if __name__ == "__main__":
    main()