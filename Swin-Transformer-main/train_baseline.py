#!/usr/bin/env python3
"""
基线分类：普通多层 CNN 与 ResNet50，与 train_simple.py (Swin) 对照。
数据集、增强、训练设置与 train_simple 保持一致。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import Counter

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# 与 train_simple 相同的 ImageNet 归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_class_weights(dataset):
    """与 train_simple 一致：类别权重"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = len(targets)
    n_cls = len(class_counts)
    weights = [total / (n_cls * class_counts[i]) for i in sorted(class_counts.keys())]
    return torch.FloatTensor(weights), class_counts


def get_transforms(augmentation_level):
    """与 train_simple 一致的增强级别：basic/medium/enhanced/strong"""
    if augmentation_level == 'basic':
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif augmentation_level == 'medium':
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif augmentation_level == 'enhanced':
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif augmentation_level == 'strong':
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        raise ValueError(f"augmentation_level: {augmentation_level}")
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_simple_cnn(num_classes):
    """普通多层 CNN：4 个卷积块 + 全连接"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 112
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 28
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    return SimpleCNN(num_classes=num_classes)


def build_resnet50(num_classes, pretrained=True):
    """ResNet50，替换最后一层为 num_classes"""
    if pretrained:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_data_loaders(data_path, batch_size, num_workers, augmentation_level):
    train_tf, val_tf = get_transforms(augmentation_level)
    train_ds = ImageFolder(os.path.join(data_path, 'train'), transform=train_tf)
    val_ds = ImageFolder(os.path.join(data_path, 'val'), transform=val_tf)
    class_weights, class_counts = compute_class_weights(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes, class_weights, class_counts


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for data, target in tqdm(loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, pred = out.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)
            loss_sum += loss.item()
            _, pred = out.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='基线分类：CNN / ResNet50')
    parser.add_argument('--data-path', type=str, required=True, help='数据集根目录（含 train/val）')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet50'], help='cnn 或 resnet50')
    parser.add_argument('--output', type=str, default='output/baseline_cnn', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--augmentation', type=str, default='enhanced', choices=['basic', 'medium', 'enhanced', 'strong'])
    parser.add_argument('--resnet-no-pretrain', action='store_true', help='ResNet50 不加载 ImageNet 预训练')
    parser.add_argument('--no-mlflow', action='store_true', help='禁用 MLflow')
    parser.add_argument('--mlflow-experiment', type=str, default='baseline-classification')
    parser.add_argument('--mlflow-run-name', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_mlflow = MLFLOW_AVAILABLE and not args.no_mlflow
    os.makedirs(args.output, exist_ok=True)

    # 数据
    print("📊 创建数据加载器...")
    train_loader, val_loader, class_names, class_weights, class_counts = create_data_loaders(
        args.data_path, args.batch_size, args.num_workers, args.augmentation
    )
    num_classes = len(class_names)
    print(f"  训练: {len(train_loader.dataset)}, 验证: {len(val_loader.dataset)}, 类别: {num_classes}")

    # 模型
    if args.model == 'cnn':
        model = build_simple_cnn(num_classes)
        default_exp = 'baseline-cnn'
    else:
        model = build_resnet50(num_classes, pretrained=not args.resnet_no_pretrain)
        default_exp = 'baseline-resnet50'
    if use_mlflow:
        exp = args.mlflow_experiment if args.mlflow_experiment != 'baseline-classification' else default_exp
        mlflow_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlruns')
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f'file://{os.path.abspath(mlflow_dir)}')
        mlflow.set_experiment(exp)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params({
            'model': args.model, 'data_path': args.data_path, 'output': args.output,
            'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': args.lr,
            'augmentation': args.augmentation, 'resnet_pretrained': not args.resnet_no_pretrain,
        })
        print(f"📈 MLflow: {exp}")

    model = model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"  训练 Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | 验证 Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | lr: {lr:.6f}")

        if use_mlflow:
            mlflow.log_metrics({
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc, 'learning_rate': lr,
            }, step=epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc, 'val_acc': val_acc, 'train_loss': train_loss, 'val_loss': val_loss,
                'class_names': class_names,
            }
            path = os.path.join(args.output, 'best_model.pth')
            torch.save(ckpt, path)
            print(f"💾 最佳模型: {path} (Acc: {val_acc:.2f}%)")
            if use_mlflow:
                mlflow.log_artifact(path, artifact_path='models')
                mlflow.log_metric('best_val_acc', val_acc, step=epoch + 1)

        if (epoch + 1) % 10 == 0:
            ckpt_period = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc, 'val_acc': val_acc, 'train_loss': train_loss, 'val_loss': val_loss,
                'class_names': class_names,
            }
            torch.save(ckpt_period, os.path.join(args.output, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"💾 检查点: checkpoint_epoch_{epoch+1}.pth")

    print(f"\n🎉 完成! 最佳验证准确率: {best_acc:.2f}%")
    if use_mlflow:
        mlflow.log_metric('best_val_acc', best_acc)
        mlflow.end_run()
        print("📈 MLflow 已记录，运行 mlflow ui 查看")


if __name__ == '__main__':
    main()
