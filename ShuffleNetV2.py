

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torchvision.models.shufflenetv2 import (
    ShuffleNet_V2_X2_0_Weights,
    shufflenet_v2_x2_0,
)

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

LOGGER_NAME = "train_logger"



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
        return transforms.Compose(
            [
                transforms.Resize(224),  # 短边 resize 到 384
                transforms.CenterCrop(224),  # 中心裁剪 384x384
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        try:
            image = Image.open(record.path).convert("RGB")
        except Exception as e:
            logging.getLogger(LOGGER_NAME).warning("Skipping broken image %s: %s", record.path, e)
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
        weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    except AttributeError:
        weights = None
    model = shufflenet_v2_x2_0(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _to_logits(outputs: Union[torch.Tensor, object]) -> torch.Tensor:
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


def _set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)

def _setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _class_distribution(samples: Sequence[SampleRecord], attr: str) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for record in samples:
        label = getattr(record, attr)
        counts[label] = counts.get(label, 0) + 1
    return counts



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="包含 images/ 和 annotation of images/ 的根路径",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="log file path (default: <root>/logs/train.log)",
    )
    parser.add_argument("--debug-samples", action="store_true", help="print first 10 samples and exit")
    parser.add_argument(
        "--target-field",
        choices=list(TARGET_FIELD_ATTRS.keys()),
        default="retinopathy",
        help="训练/验证时使用的目标字段(macular_edema 或 retinopathy)",
    )
    args = parser.parse_args()

    log_file = args.log_file or (args.root / "logs" / "train.log")
    logger = _setup_logger(log_file)
    logger.info("Log file: %s", log_file)

    generator = _set_seed(args.seed)
    image_root = args.root / "images"
    annotation_root = args.root / "annotation of images"
    dataset = RetinaDataset(image_root, annotation_root)
    target_attr = TARGET_FIELD_ATTRS[args.target_field]
    target_values = [getattr(record, target_attr) for record in dataset.samples]
    if not target_values:
        raise RuntimeError("样本列表为空，无法确定类别数")
    num_classes = max(target_values) + 1
    logger.info("Target field: %s", args.target_field)
    logger.info("Dataset size: %d", len(dataset))
    logger.info("Class distribution: %s", _class_distribution(dataset.samples, target_attr))

    val_count = int(len(dataset) * args.val_ratio)
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)
    logger.info("Split: train=%d val=%d (val_ratio=%.2f)", train_count, val_count, args.val_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    total_params, trainable_params = _count_params(model)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
    logger.info("Model: %s", model.__class__.__name__)
    logger.info("Params: total=%d trainable=%d", total_params, trainable_params)
    logger.info("Model architecture:\n%s", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Optimizer: %s lr=%.6f", optimizer.__class__.__name__, args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info("Batch size: %d", args.batch_size)
    if args.debug_samples:
        for record in dataset.samples[:10]:
            logger.info("Sample: %s", record)
        dist = _class_distribution(dataset.samples, "retinopathy_grade")
        logger.info("Sample count=%d, class distribution=%s", len(dataset), dist)
        return

    history: List[Dict[str, float]] = []
    logger.info("Training start: epochs=%d", args.epochs)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            target_key=args.target_field,
        )
        acc = evaluate(model, val_loader, device, target_key=args.target_field)
        history.append({"epoch": epoch, "loss": loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"]})
        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d: loss=%.4f val_acc=%.4f lr=%.6f time=%.1fs",
            epoch,
            args.epochs,
            loss,
            acc,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

    for record in history:
        logger.info(
            " epoch=%d loss=%.4f val_acc=%.4f lr=%.6f",
            int(record["epoch"]),
            record["loss"],
            record["val_acc"],
            record["lr"],
        )

    final_counts = evaluate_with_counts(model, val_loader, device, target_key=args.target_field)
    for label, total in sorted(final_counts["total"].items()):
        correct = final_counts["correct"].get(label, 0)
        pct = correct / total if total else 0
        logger.info("  class %d: %d/%d (%.2f%%)", label, correct, total, pct * 100)


if __name__ == "__main__":
    main()
