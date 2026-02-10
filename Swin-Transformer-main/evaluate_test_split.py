#!/usr/bin/env python3
"""评估模型在 data/full_dataset/test/ 上的分类准确率"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
import sys

sys.path.append(str(PROJECT_ROOT))

from config import get_config  # noqa: E402
from models import build_model  # noqa: E402


def load_class_names(class_file: Path) -> List[str]:
    if not class_file.exists():
        raise FileNotFoundError(f"找不到类别名称文件: {class_file}")
    with open(class_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError("class_names.txt 为空")
    return names


def load_model(checkpoint: Path, class_names: List[str]) -> torch.nn.Module:
    print(f"📥 加载模型: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    class Args:
        def __init__(self) -> None:
            self.cfg = 'configs/swin/swin_tiny_triple_class.yaml'
            self.opts = None
            self.batch_size = None
            self.data_path = None
            self.zip = False
            self.cache_mode = None
            self.pretrained = None
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.disable_amp = False
            self.amp_opt_level = None
            self.output = None
            self.tag = None
            self.eval = False
            self.throughput = False
            self.local_rank = 0
            self.fused_window_process = False
            self.fused_layernorm = False
            self.optim = None

    os.environ['LOCAL_RANK'] = '0'
    args = Args()
    config = get_config(args)
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()

    model = build_model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("✅ 模型加载成功")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    return model


def build_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_test_images(test_dir: Path) -> Dict[str, List[Path]]:
    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    class_to_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}])
        if not images:
            continue
        class_to_images[class_dir.name] = images
    if not class_to_images:
        raise ValueError("测试集目录中未找到任何图片")
    return class_to_images


def evaluate(model: torch.nn.Module, class_names: List[str], class_to_images: Dict[str, List[Path]], device: str) -> Dict[str, object]:
    transform = build_transform()
    idx_map = {name: idx for idx, name in enumerate(class_names)}

    total = 0
    correct = 0
    per_class = {name: {"total": 0, "correct": 0} for name in class_names}

    model.to(device)

    for class_name, images in class_to_images.items():
        if class_name not in idx_map:
            print(f"⚠️  类别 {class_name} 不在模型类别列表中，已跳过")
            continue
        target_idx = idx_map[class_name]
        for image_path in images:
            with Image.open(image_path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = int(probs.argmax(dim=1).item())

            total += 1
            per_class[class_name]['total'] += 1
            if pred_idx == target_idx:
                correct += 1
                per_class[class_name]['correct'] += 1

    per_class_stats = []
    for name in class_names:
        stats = per_class.get(name, {"total": 0, "correct": 0})
        total_cls = stats['total']
        acc = stats['correct'] / total_cls if total_cls else 0.0
        per_class_stats.append({
            "class_name": name,
            "total": total_cls,
            "correct": stats['correct'],
            "accuracy": acc,
        })

    overall_acc = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": overall_acc,
        "per_class": per_class_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser("评估模型在测试集上的表现")
    parser.add_argument("--data-root", type=Path, default=Path("/root/github/Swin-Transformer-main/data/full_dataset"), help="数据集根目录 (包含 train/ val/ test/)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型权重路径")
    parser.add_argument("--class-names", type=Path, default=None, help="类别名称文件 (默认 data_root/class_names.txt)")
    parser.add_argument("--output", type=Path, default=None, help="可选：保存结果到 JSON 文件")

    args = parser.parse_args()

    class_file = args.class_names or (args.data_root / "class_names.txt")
    class_names = load_class_names(class_file)

    model = load_model(args.checkpoint, class_names)

    test_dir = args.data_root / "test"
    class_to_images = collect_test_images(test_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = evaluate(model, class_names, class_to_images, device)

    print("\n" + "=" * 70)
    print("✅ 测试集评估完成")
    print("=" * 70)
    print(f"总样本数: {report['total']}")
    print(f"正确数量: {report['correct']}")
    print(f"总体准确率: {report['accuracy'] * 100:.2f}%")

    print("\n按类别统计:")
    for item in report['per_class']:
        print(
            f"  - {item['class_name']}: {item['correct']}/{item['total']} "
            f"({item['accuracy'] * 100:.2f}%)"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()


