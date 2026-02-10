#!/usr/bin/env python3
"""
简化的Swin-Transformer训练脚本
避免分布式训练的复杂性，直接使用单GPU训练
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import numpy as np
from collections import Counter

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# 添加项目路径
sys.path.append('/root/github/Swin-Transformer-main')

from config import get_config
from models import build_model

def compute_class_weights(dataset):
    """计算类别权重（用于处理不平衡数据集）"""
    # 统计每个类别的样本数量
    targets = [sample[1] for sample in dataset.samples]
    class_counts = Counter(targets)
    
    # 计算总样本数
    total_samples = len(targets)
    num_classes = len(class_counts)
    
    # 计算类别权重：weight = total / (num_classes * class_count)
    class_weights = []
    for class_idx in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_idx])
        class_weights.append(weight)
    
    return torch.FloatTensor(class_weights), class_counts

def create_data_loaders(data_path, batch_size=32, num_workers=4, augmentation_level='enhanced'):
    """创建数据加载器
    
    Args:
        augmentation_level: 'basic', 'medium', 'enhanced', 'strong'
    """
    
    # 数据预处理 - 根据增强级别选择
    if augmentation_level == 'basic':
        # 基础增强
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'medium':
        # 中等增强（原版）
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'enhanced':
        # 增强版（推荐）
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),                    # 先放大
            transforms.RandomCrop((224, 224)),                # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),           # 水平翻转
            transforms.RandomVerticalFlip(p=0.3),             # 垂直翻转
            transforms.RandomRotation(degrees=20),            # 旋转±20度
            transforms.RandomAffine(                          # 仿射变换
                degrees=0,
                translate=(0.1, 0.1),                         # 平移10%
                scale=(0.9, 1.1),                             # 缩放90%-110%
                shear=5                                        # 剪切5度
            ),
            transforms.ColorJitter(                           # 颜色抖动
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15
            ),
            transforms.RandomGrayscale(p=0.05),               # 5%概率转灰度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_level == 'strong':
        # 强增强
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 透视变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown augmentation_level: {augmentation_level}")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_path, 'val'), transform=val_transform)
    
    # 计算类别权重
    class_weights, class_counts = compute_class_weights(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, train_dataset.classes, class_weights, class_counts

def load_pretrained_weights(model, pretrained_path, num_classes=3):
    """加载预训练权重"""
    print(f"📥 加载预训练权重: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 过滤掉分类头权重
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"✅ 预训练权重加载成功")
    print(f"   跳过的分类头权重: {len(state_dict) - len(filtered_state_dict)} 个")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser('简化Swin-Transformer训练')
    parser.add_argument('--data-path', type=str, required=True, help='数据集路径')
    parser.add_argument('--pretrained', type=str, default='pretrain/swin_tiny_patch4_window7_224.pth', help='预训练权重路径')
    parser.add_argument('--output', type=str, default='output/simple_training', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--augmentation', type=str, default='enhanced', 
                       choices=['basic', 'medium', 'enhanced', 'strong'],
                       help='数据增强级别: basic/medium/enhanced/strong')
    parser.add_argument('--no-mlflow', action='store_true', help='禁用 MLflow 监控')
    parser.add_argument('--mlflow-experiment', type=str, default='swin-simple-training',
                       help='MLflow 实验名称')
    parser.add_argument('--mlflow-run-name', type=str, default=None,
                       help='MLflow 运行名称（默认自动生成）')
    
    args = parser.parse_args()
    
    use_mlflow = MLFLOW_AVAILABLE and not args.no_mlflow
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # MLflow 监控
    if use_mlflow:
        mlflow_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlruns')
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f'file://{os.path.abspath(mlflow_dir)}')
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params({
            'data_path': args.data_path,
            'pretrained': args.pretrained,
            'output': args.output,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'num_workers': args.num_workers,
            'augmentation': args.augmentation,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
        })
        print(f"📈 MLflow 监控已开启 (experiment: {args.mlflow_experiment})")
    elif not MLFLOW_AVAILABLE:
        print("⚠️  未安装 mlflow，跳过监控。安装: pip install mlflow")
    
    # 创建数据加载器
    print("📊 创建数据加载器...")
    print(f"📸 数据增强级别: {args.augmentation}")
    train_loader, val_loader, class_names, class_weights, class_counts = create_data_loaders(
        args.data_path, args.batch_size, args.num_workers, args.augmentation
    )
    
    print(f"📋 类别信息:")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        weight = class_weights[i].item()
        print(f"  {i}: {name} - {count}张 (权重: {weight:.4f})")
    
    print(f"📊 数据集统计:")
    print(f"  训练集: {len(train_loader.dataset)} 个样本")
    print(f"  验证集: {len(val_loader.dataset)} 个样本")
    
    # 创建模型
    print("🏗️  创建模型...")
    
    # 创建配置
    class Args:
        def __init__(self):
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
    
    # 设置环境变量
    os.environ['LOCAL_RANK'] = '0'
    
    config_args = Args()
    config = get_config(config_args)
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()
    
    model = build_model(config)
    model = model.to(device)
    
    # 加载预训练权重
    if os.path.exists(args.pretrained):
        model = load_pretrained_weights(model, args.pretrained, len(class_names))
    else:
        print(f"⚠️  预训练权重文件不存在: {args.pretrained}")
        print("   将从头开始训练")
    
    # 创建优化器和损失函数（带类别权重）
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n⚖️  类别权重已应用:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {name}: {weight.item():.4f}")
    
    print(f"🔧 训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  优化器: AdamW")
    print(f"  调度器: CosineAnnealingLR")
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        print(f"📊 结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # MLflow 记录每轮指标
        if use_mlflow:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'class_names': class_names
            }
            
            best_model_path = os.path.join(args.output, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"💾 保存最佳模型: {best_model_path} (Acc: {val_acc:.2f}%)")
            if use_mlflow:
                mlflow.log_artifact(best_model_path, artifact_path='models')
                mlflow.log_metric('best_val_acc', val_acc, step=epoch + 1)
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint['epoch'] = epoch
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
    
    print(f"\n🎉 训练完成!")
    print(f"🏆 最佳验证准确率: {best_acc:.2f}%")
    print(f"📁 模型保存在: {args.output}")
    
    if use_mlflow:
        mlflow.log_metric('best_val_acc', best_acc)
        mlflow.end_run()
        print(f"📈 MLflow 运行已结束，可在 mlruns 目录查看或运行: mlflow ui")

if __name__ == '__main__':
    main()
