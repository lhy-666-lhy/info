#!/usr/bin/env python3
"""
数据集准备脚本
将带标注的原始数据按 6:2:2 划分为训练/验证/测试集
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import json
import xml.etree.ElementTree as ET

# 设置随机种子以确保可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 数据集配置
SOURCE_ROOT = '/root/github/图像与标注'
TARGET_ROOT = '/root/github/Swin-Transformer-main/data/full_dataset'

# 类别映射
CLASS_MAPPING = {
    '产生血小板型巨核细胞': 'class0_platelet_producing',
    '巨核细胞裸核': 'class1_bare_nucleus',
    '颗粒型巨核细胞': 'class2_granular'
}

# 划分比例 (train : val : test = 6 : 2 : 2)
SPLIT_RATIOS = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2,
}

IMAGE_EXTENSIONS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']


def collect_images(directory: Path):
    """搜集指定目录下的所有图像文件"""
    search_dir = directory / 'images'
    if not search_dir.exists():
        search_dir = directory

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(search_dir.glob(f'*{ext}'))
        images.extend(search_dir.glob(f'*{ext.upper()}'))

    return sorted(set(images))

def prepare_balanced_dataset():
    """按照给定比例准备数据集"""
    
    print("="*70)
    ratio_text = f"{int(SPLIT_RATIOS['train']*10)}:{int(SPLIT_RATIOS['val']*10)}:{int(SPLIT_RATIOS['test']*10)}"
    print(f"🎯 数据集准备 - 按 {ratio_text} 划分 train/val/test")
    print("="*70)
    
# 1. 解析 XML 标注，确定每张图片的类别
    print("\n📊 解析 XML 标注...")
    xml_dir = Path(SOURCE_ROOT) / 'xml'
    image_dir = Path(SOURCE_ROOT) / '原图'
    if not xml_dir.exists() or not image_dir.exists():
        raise FileNotFoundError("请确保在 SOURCE_ROOT 下存在 'xml/' 和 '原图/' 目录")

    class_images = defaultdict(list)
    missing_images = []
    unresolved_labels = []

    for xml_file in sorted(xml_dir.glob('*.xml')):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  无法解析 {xml_file.name}: {exc}")
            continue

        filename_node = root.find('filename')
        if filename_node is None or not filename_node.text:
            print(f"⚠️  {xml_file.name} 缺少 filename 字段，已跳过")
            continue

        image_name = filename_node.text.strip()
        image_path = image_dir / image_name
        if not image_path.exists():
            missing_images.append(image_path)
            continue

        labels = []
        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is not None and name_node.text:
                labels.append(name_node.text.strip())

        if not labels:
            print(f"⚠️  {xml_file.name} 中未找到任何 object/name，已跳过")
            continue

        # 使用出现次数最多的类别
        majority_label = max(labels, key=labels.count)
        if majority_label not in CLASS_MAPPING:
            unresolved_labels.append((xml_file.name, majority_label))
            continue

        class_id = CLASS_MAPPING[majority_label]
        class_images[class_id].append(image_path)

    if unresolved_labels:
        print(f"⚠️  有 {len(unresolved_labels)} 个标签无法映射，将跳过示例:")
        for xml_name, label in unresolved_labels[:5]:
            print(f"    - {xml_name}: {label}")

    if missing_images:
        print(f"⚠️  有 {len(missing_images)} 张图片缺失，将跳过示例:")
        for path in missing_images[:5]:
            print(f"    - {path}")

    if not class_images:
        raise ValueError("未从 XML 中解析到任何有效的图像和类别")

    class_counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    for chinese, english in CLASS_MAPPING.items():
        print(f"  {chinese} -> {english}: {class_counts.get(english, 0)} 张")

    # 2. 划分数据集
    print("\n📂 划分训练/验证/测试集...")
    dataset_split = {split: defaultdict(list) for split in ['train', 'val', 'test']}
    split_totals = {split: 0 for split in dataset_split}

    for class_name, images in class_images.items():
        random.shuffle(images)
        total = len(images)
        train_count = int(total * SPLIT_RATIOS['train'])
        val_count = int(total * SPLIT_RATIOS['val'])
        test_count = total - train_count - val_count

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        dataset_split['train'][class_name] = train_images
        dataset_split['val'][class_name] = val_images
        dataset_split['test'][class_name] = test_images

        split_totals['train'] += len(train_images)
        split_totals['val'] += len(val_images)
        split_totals['test'] += len(test_images)

        print(
            f"  {class_name}: 训练 {len(train_images)} 张, 验证 {len(val_images)} 张, 测试 {len(test_images)} 张"
        )

    # 3. 创建目标目录并复制文件
    print(f"\n📁 创建数据集目录: {TARGET_ROOT}")
    
    # 清空旧数据（如果存在）
    if os.path.exists(TARGET_ROOT):
        print("  ⚠️  删除旧数据集...")
        shutil.rmtree(TARGET_ROOT)
    
    # 统计信息
    stats = {
        'total_images': sum(class_counts.values()),
        'split_totals': split_totals,
        'split_ratios': SPLIT_RATIOS,
        'random_seed': RANDOM_SEED,
        'classes': {},
        'class_mapping': CLASS_MAPPING
    }
    
    # 5. 复制文件
    print("\n📋 复制文件...")
    for split, class_dict in dataset_split.items():
        print(f"\n  {split.upper()}:")
        for class_name, images in class_dict.items():
            class_dir = Path(TARGET_ROOT) / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            copied_count = 0
            for img_path in images:
                dest_path = class_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                copied_count += 1

            print(f"    {class_name}: {copied_count} 张 ✓")
            stats['classes'].setdefault(class_name, {})[split] = copied_count
    
    # 6. 保存数据集信息
    info_file = os.path.join(TARGET_ROOT, 'dataset_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 数据集信息已保存: {info_file}")
    
    # 7. 创建类别名称文件（用于训练）
    class_names = list(CLASS_MAPPING.values())
    class_names_file = os.path.join(TARGET_ROOT, 'class_names.txt')
    with open(class_names_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"📝 类别名称已保存: {class_names_file}")
    
    # 8. 打印总结
    print("\n" + "="*70)
    print("✅ 数据集准备完成！")
    print("="*70)
    print(f"\n📊 数据集统计:")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  类别数: {len(class_names)}")
    for split, total in split_totals.items():
        print(f"  {split} 集总数: {total} 张")

    print(f"\n📁 数据集位置: {TARGET_ROOT}")
    for split in ['train', 'val', 'test']:
        print(f"  - {split}/")
        for class_name in class_names:
            count = stats['classes'].get(class_name, {}).get(split, 0)
            print(f"    - {class_name}/ ({count} 张)")
    
    print("\n" + "="*70)
    
    return TARGET_ROOT, stats

def verify_dataset(dataset_root):
    """验证数据集"""
    print("\n🔍 验证数据集...")
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"  ❌ {split} 目录不存在")
            return False
        
        class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(Path(class_dir).glob(f'*{ext}'))
                images.extend(Path(class_dir).glob(f'*{ext.upper()}'))
            print(f"  ✓ {split}/{class_name}: {len(set(images))} 张")
    
    print("✅ 数据集验证完成")
    return True

if __name__ == '__main__':
    try:
        # 准备数据集
        dataset_root, stats = prepare_balanced_dataset()
        
        # 验证数据集
        verify_dataset(dataset_root)
        
        print("\n🎉 一切就绪！现在可以开始训练了。")
        print(f"\n运行训练命令:")
        print(f"  python train_simple.py --data-path {dataset_root} --epochs 100 --batch-size 32")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

