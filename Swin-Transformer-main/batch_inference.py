#!/usr/bin/env python3
"""
批量推理脚本 - 用于测试模型在特定类别上的准确率
支持统计准确率、置信度分布、混淆情况等
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json

# 添加项目路径
sys.path.append('/root/github/Swin-Transformer-main')

from config import get_config
from models import build_model


class BatchInferenceEvaluator:
    """批量推理评估器"""
    
    def __init__(self, model, class_names, true_label_idx, img_size=224):
        self.model = model
        self.class_names = class_names
        self.true_label_idx = true_label_idx
        self.img_size = img_size
        
        # 统计信息
        self.results = {
            'correct': 0,
            'total': 0,
            'predictions': [],  # 记录每张图片的预测结果
            'confidences': [],  # 记录所有预测的置信度
            'true_class_probs': [],  # 记录真实类别的概率
            'confusion': {i: 0 for i in range(len(class_names))},  # 混淆统计
            'failed_images': []  # 预测错误的图片
        }
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor
        except Exception as e:
            print(f"  ⚠️  图片加载失败 {image_path}: {e}")
            return None
    
    def predict_single(self, image_path):
        """预测单张图片"""
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return {
                'predicted_class': predicted.item(),
                'confidence': confidence.item(),
                'all_probs': probabilities[0].cpu().numpy(),
                'true_class_prob': probabilities[0][self.true_label_idx].item()
            }
    
    def evaluate_batch(self, image_folder):
        """批量评估文件夹中的所有图片"""
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        
        if len(image_files) == 0:
            print(f"❌ 在 {image_folder} 中未找到图片文件")
            return
        
        print(f"📊 找到 {len(image_files)} 张图片")
        print(f"🎯 真实类别: {self.class_names[self.true_label_idx]}")
        print("="*70)
        print("\n开始批量推理...\n")
        
        # 使用tqdm显示进度条
        for image_path in tqdm(image_files, desc="推理进度", ncols=100):
            result = self.predict_single(str(image_path))
            
            if result is None:
                continue
            
            self.results['total'] += 1
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            true_class_prob = result['true_class_prob']
            
            # 记录预测结果
            self.results['predictions'].append({
                'image': image_path.name,
                'predicted_class': predicted_class,
                'predicted_name': self.class_names[predicted_class],
                'confidence': confidence,
                'true_class_prob': true_class_prob,
                'correct': predicted_class == self.true_label_idx
            })
            
            self.results['confidences'].append(confidence)
            self.results['true_class_probs'].append(true_class_prob)
            self.results['confusion'][predicted_class] += 1
            
            # 判断是否预测正确
            if predicted_class == self.true_label_idx:
                self.results['correct'] += 1
            else:
                self.results['failed_images'].append({
                    'image': image_path.name,
                    'predicted_class': self.class_names[predicted_class],
                    'confidence': confidence,
                    'true_class_prob': true_class_prob
                })
        
        print("\n" + "="*70)
        print("✅ 批量推理完成！")
        print("="*70)
    
    def get_statistics(self):
        """获取统计信息"""
        if self.results['total'] == 0:
            return None
        
        accuracy = self.results['correct'] / self.results['total']
        
        stats = {
            'accuracy': accuracy,
            'total': self.results['total'],
            'correct': self.results['correct'],
            'wrong': self.results['total'] - self.results['correct'],
            'avg_confidence': np.mean(self.results['confidences']),
            'std_confidence': np.std(self.results['confidences']),
            'avg_true_class_prob': np.mean(self.results['true_class_probs']),
            'std_true_class_prob': np.std(self.results['true_class_probs']),
            'min_confidence': np.min(self.results['confidences']),
            'max_confidence': np.max(self.results['confidences']),
            'confusion': self.results['confusion']
        }
        
        return stats
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_statistics()
        if stats is None:
            print("❌ 没有可用的统计数据")
            return
        
        print("\n" + "="*70)
        print("📊 评估统计摘要")
        print("="*70)
        print(f"\n总体结果:")
        print(f"  总图片数: {stats['total']}")
        print(f"  预测正确: {stats['correct']} ✅")
        print(f"  预测错误: {stats['wrong']} ❌")
        print(f"  准确率: {stats['accuracy']*100:.2f}%")
        
        # 准确率可视化条
        correct_bar = '█' * int(stats['accuracy'] * 50)
        wrong_bar = '░' * (50 - int(stats['accuracy'] * 50))
        print(f"\n  {correct_bar}{wrong_bar} {stats['accuracy']*100:.1f}%")
        
        print(f"\n置信度统计:")
        print(f"  平均置信度: {stats['avg_confidence']:.4f} (±{stats['std_confidence']:.4f})")
        print(f"  最小置信度: {stats['min_confidence']:.4f}")
        print(f"  最大置信度: {stats['max_confidence']:.4f}")
        
        print(f"\n真实类别概率统计:")
        print(f"  平均概率: {stats['avg_true_class_prob']:.4f} (±{stats['std_true_class_prob']:.4f})")
        
        print(f"\n混淆分布 (预测为各类别的数量):")
        for class_idx, count in stats['confusion'].items():
            percentage = count / stats['total'] * 100
            bar = '█' * int(percentage / 2)
            marker = '✅' if class_idx == self.true_label_idx else '❌'
            print(f"  {marker} {self.class_names[class_idx]}: {count} ({percentage:.1f}%) {bar}")
        
        if len(self.results['failed_images']) > 0:
            print(f"\n❌ 预测错误的图片 (共{len(self.results['failed_images'])}张):")
            for i, fail in enumerate(self.results['failed_images'][:10], 1):
                print(f"  {i}. {fail['image']}")
                print(f"     预测为: {fail['predicted_class']} (置信度: {fail['confidence']:.4f})")
                print(f"     真实类别概率: {fail['true_class_prob']:.4f}")
            
            if len(self.results['failed_images']) > 10:
                print(f"  ... 还有 {len(self.results['failed_images']) - 10} 张错误图片")
        
        print("="*70)


def plot_statistics(evaluator, output_dir):
    """绘制统计图表"""
    stats = evaluator.get_statistics()
    if stats is None:
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. 准确率饼图
    ax1 = plt.subplot(2, 3, 1)
    sizes = [stats['correct'], stats['wrong']]
    labels = [f"正确 ({stats['correct']})", f"错误 ({stats['wrong']})"]
    colors = ['#4CAF50', '#F44336']
    explode = (0.05, 0)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11})
    ax1.set_title(f"预测准确率: {stats['accuracy']*100:.2f}%", fontsize=14, pad=20)
    
    # 2. 混淆分布条形图
    ax2 = plt.subplot(2, 3, 2)
    class_names = evaluator.class_names
    confusion_values = [stats['confusion'][i] for i in range(len(class_names))]
    colors_bar = ['#4CAF50' if i == evaluator.true_label_idx else '#2196F3' 
                  for i in range(len(class_names))]
    bars = ax2.bar(class_names, confusion_values, color=colors_bar, alpha=0.7)
    ax2.set_xlabel('预测类别', fontsize=11)
    ax2.set_ylabel('数量', fontsize=11)
    ax2.set_title('预测类别分布', fontsize=14, pad=20)
    ax2.grid(axis='y', alpha=0.3)
    
    # 在条形图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 3. 置信度分布直方图
    ax3 = plt.subplot(2, 3, 3)
    confidences = evaluator.results['confidences']
    ax3.hist(confidences, bins=20, color='#673AB7', alpha=0.7, edgecolor='black')
    ax3.axvline(stats['avg_confidence'], color='red', linestyle='--', 
                linewidth=2, label=f"平均值: {stats['avg_confidence']:.3f}")
    ax3.set_xlabel('置信度', fontsize=11)
    ax3.set_ylabel('数量', fontsize=11)
    ax3.set_title('置信度分布', fontsize=14, pad=20)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 真实类别概率分布
    ax4 = plt.subplot(2, 3, 4)
    true_probs = evaluator.results['true_class_probs']
    ax4.hist(true_probs, bins=20, color='#FF9800', alpha=0.7, edgecolor='black')
    ax4.axvline(stats['avg_true_class_prob'], color='red', linestyle='--',
                linewidth=2, label=f"平均值: {stats['avg_true_class_prob']:.3f}")
    ax4.set_xlabel(f'真实类别({class_names[evaluator.true_label_idx]})概率', fontsize=11)
    ax4.set_ylabel('数量', fontsize=11)
    ax4.set_title('真实类别概率分布', fontsize=14, pad=20)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 置信度箱线图
    ax5 = plt.subplot(2, 3, 5)
    correct_confidences = [p['confidence'] for p in evaluator.results['predictions'] if p['correct']]
    wrong_confidences = [p['confidence'] for p in evaluator.results['predictions'] if not p['correct']]
    
    box_data = []
    box_labels = []
    if len(correct_confidences) > 0:
        box_data.append(correct_confidences)
        box_labels.append('正确预测')
    if len(wrong_confidences) > 0:
        box_data.append(wrong_confidences)
        box_labels.append('错误预测')
    
    if len(box_data) > 0:
        bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors_box = ['#4CAF50', '#F44336']
        for patch, color in zip(bp['boxes'], colors_box[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax5.set_ylabel('置信度', fontsize=11)
    ax5.set_title('正确vs错误预测的置信度对比', fontsize=14, pad=20)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. 统计摘要文本
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    评估统计摘要
    {'='*35}
    
    总图片数: {stats['total']}
    预测正确: {stats['correct']}
    预测错误: {stats['wrong']}
    准确率: {stats['accuracy']*100:.2f}%
    
    置信度:
      平均: {stats['avg_confidence']:.4f}
      标准差: {stats['std_confidence']:.4f}
      范围: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]
    
    真实类别概率:
      平均: {stats['avg_true_class_prob']:.4f}
      标准差: {stats['std_true_class_prob']:.4f}
    
    真实类别: {class_names[evaluator.true_label_idx]}
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'evaluation_statistics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 统计图表已保存: {plot_path}")


def save_detailed_report(evaluator, output_dir, image_folder, checkpoint_path):
    """保存详细评估报告"""
    stats = evaluator.get_statistics()
    if stats is None:
        return
    
    report_path = os.path.join(output_dir, 'batch_evaluation_report.md')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 📊 批量推理评估报告\n\n")
        f.write(f"**生成时间**: {timestamp}\n\n")
        f.write("---\n\n")
        
        # 基本信息
        f.write("## 📋 评估信息\n\n")
        f.write(f"- **图像文件夹**: `{image_folder}`\n")
        f.write(f"- **模型路径**: `{checkpoint_path}`\n")
        f.write(f"- **真实类别**: `{evaluator.class_names[evaluator.true_label_idx]}`\n")
        f.write(f"- **图片总数**: {stats['total']}\n\n")
        
        # 准确率结果
        f.write("## 🎯 准确率结果\n\n")
        f.write(f"### 总体准确率: **{stats['accuracy']*100:.2f}%**\n\n")
        
        # 准确率进度条
        bar_length = int(stats['accuracy'] * 50)
        accuracy_bar = '█' * bar_length + '░' * (50 - bar_length)
        f.write(f"```\n{accuracy_bar} {stats['accuracy']*100:.1f}%\n```\n\n")
        
        f.write(f"- ✅ **正确预测**: {stats['correct']} 张 ({stats['correct']/stats['total']*100:.1f}%)\n")
        f.write(f"- ❌ **错误预测**: {stats['wrong']} 张 ({stats['wrong']/stats['total']*100:.1f}%)\n\n")
        
        # 置信度统计
        f.write("## 📈 置信度统计\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 平均置信度 | {stats['avg_confidence']:.4f} |\n")
        f.write(f"| 标准差 | {stats['std_confidence']:.4f} |\n")
        f.write(f"| 最小置信度 | {stats['min_confidence']:.4f} |\n")
        f.write(f"| 最大置信度 | {stats['max_confidence']:.4f} |\n")
        f.write(f"| 真实类别平均概率 | {stats['avg_true_class_prob']:.4f} |\n")
        f.write(f"| 真实类别概率标准差 | {stats['std_true_class_prob']:.4f} |\n\n")
        
        # 混淆分布
        f.write("## 📊 预测类别分布\n\n")
        f.write("| 预测类别 | 数量 | 百分比 | 可视化 |\n")
        f.write("|----------|------|--------|--------|\n")
        
        for class_idx, count in stats['confusion'].items():
            percentage = count / stats['total'] * 100
            bar = '█' * int(percentage / 2) + '░' * (50 - int(percentage / 2))
            marker = '✅' if class_idx == evaluator.true_label_idx else '❌'
            f.write(f"| {evaluator.class_names[class_idx]} {marker} | {count} | {percentage:.1f}% | `{bar}` |\n")
        
        f.write("\n")
        
        # 可视化结果
        f.write("## 🖼️ 统计可视化\n\n")
        f.write("![统计图表](evaluation_statistics.png)\n\n")
        
        # 错误预测详情
        if len(evaluator.results['failed_images']) > 0:
            f.write(f"## ❌ 预测错误详情 (共{len(evaluator.results['failed_images'])}张)\n\n")
            f.write("| 序号 | 图片文件名 | 预测类别 | 置信度 | 真实类别概率 |\n")
            f.write("|------|-----------|----------|--------|-------------|\n")
            
            for i, fail in enumerate(evaluator.results['failed_images'], 1):
                f.write(f"| {i} | `{fail['image']}` | {fail['predicted_class']} | {fail['confidence']:.4f} | {fail['true_class_prob']:.4f} |\n")
            
            f.write("\n")
        
        # 分析总结
        f.write("## 💡 分析总结\n\n")
        
        if stats['accuracy'] >= 0.95:
            f.write("✅ **模型表现优秀**：准确率超过95%，模型在该类别上表现非常好。\n\n")
        elif stats['accuracy'] >= 0.85:
            f.write("⚠️ **模型表现良好**：准确率在85%-95%之间，模型表现较好但仍有改进空间。\n\n")
        elif stats['accuracy'] >= 0.70:
            f.write("⚠️ **模型表现一般**：准确率在70%-85%之间，建议进一步优化模型。\n\n")
        else:
            f.write("❌ **模型表现较差**：准确率低于70%，需要重新训练或调整模型。\n\n")
        
        # 置信度分析
        if stats['avg_confidence'] >= 0.90:
            f.write(f"- 平均置信度较高({stats['avg_confidence']:.2%})，模型对预测结果较为确信\n")
        elif stats['avg_confidence'] >= 0.70:
            f.write(f"- 平均置信度中等({stats['avg_confidence']:.2%})，模型预测存在一定不确定性\n")
        else:
            f.write(f"- 平均置信度较低({stats['avg_confidence']:.2%})，模型对很多预测不够确信\n")
        
        # 真实类别概率分析
        if stats['avg_true_class_prob'] >= 0.80:
            f.write(f"- 真实类别平均概率较高({stats['avg_true_class_prob']:.2%})，模型能够较好识别该类别\n")
        else:
            f.write(f"- 真实类别平均概率偏低({stats['avg_true_class_prob']:.2%})，模型对该类别的识别能力需要提升\n")
        
        f.write("\n")
        
        # 技术信息
        f.write("---\n\n")
        f.write("## 🔧 技术信息\n\n")
        f.write("- **模型架构**: Swin-Transformer\n")
        f.write(f"- **类别数量**: {len(evaluator.class_names)}\n")
        f.write(f"- **输入尺寸**: {evaluator.img_size}×{evaluator.img_size}\n")
        f.write(f"- **评估时间**: {timestamp}\n\n")
        
        f.write("---\n\n")
        f.write("*此报告由批量推理脚本自动生成*\n")
    
    print(f"📄 评估报告已保存: {report_path}")
    
    # 同时保存JSON格式的详细结果
    json_path = os.path.join(output_dir, 'detailed_predictions.json')
    json_data = {
        'metadata': {
            'timestamp': timestamp,
            'image_folder': image_folder,
            'checkpoint': checkpoint_path,
            'true_label': evaluator.class_names[evaluator.true_label_idx],
            'true_label_idx': evaluator.true_label_idx
        },
        'statistics': stats,
        'predictions': evaluator.results['predictions']
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 详细结果已保存: {json_path}")


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"📥 加载模型: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取类别名称
    class_names = checkpoint.get('class_names', ['Class 0', 'Class 1', 'Class 2'])
    
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
    
    args = Args()
    config = get_config(args)
    
    # 设置类别数量
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()
    
    # 构建模型
    model = build_model(config)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"📊 类别信息:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    return model, class_names


def main():
    parser = argparse.ArgumentParser('批量推理评估脚本')
    parser.add_argument('--image-folder', type=str, required=True, 
                       help='图像文件夹路径')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='模型检查点路径')
    parser.add_argument('--true-label', type=str, default='巨核细胞裸核', 
                       help='真实类别名称')
    parser.add_argument('--true-label-idx', type=int, default=None,
                       help='真实类别索引（如果不指定，将根据类别名称自动查找）')
    parser.add_argument('--img-size', type=int, default=224, 
                       help='输入图片尺寸')
    parser.add_argument('--output-dir', type=str, default='batch_eval_result', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("="*70)
    print("📊 Swin-Transformer 批量推理评估系统")
    print("="*70)
    print(f"📁 图像文件夹: {args.image_folder}")
    print(f"💾 模型权重: {args.checkpoint}")
    print(f"🎯 真实类别: {args.true_label}")
    print(f"📁 输出目录: {args.output_dir}")
    print("="*70 + "\n")
    
    # 检查文件夹和文件是否存在
    if not os.path.exists(args.image_folder):
        print(f"❌ 图像文件夹不存在: {args.image_folder}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ 模型文件不存在: {args.checkpoint}")
        return
    
    try:
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 创建输出目录: {output_dir}\n")
        
        # 加载模型
        model, class_names = load_model(args.checkpoint)
        
        # 确定真实类别索引
        if args.true_label_idx is not None:
            true_label_idx = args.true_label_idx
        else:
            # 根据类别名称查找索引
            try:
                true_label_idx = class_names.index(args.true_label)
            except ValueError:
                print(f"⚠️  未找到类别'{args.true_label}'，使用默认值1")
                print(f"可用类别: {class_names}")
                true_label_idx = 1
        
        print(f"\n真实类别索引: {true_label_idx} ({class_names[true_label_idx]})\n")
        
        # 创建评估器
        evaluator = BatchInferenceEvaluator(model, class_names, true_label_idx, args.img_size)
        
        # 批量评估
        evaluator.evaluate_batch(args.image_folder)
        
        # 打印统计摘要
        evaluator.print_summary()
        
        # 生成可视化图表
        print("\n📊 生成统计图表...")
        plot_statistics(evaluator, output_dir)
        
        # 保存详细报告
        print("📝 生成评估报告...")
        save_detailed_report(evaluator, output_dir, args.image_folder, args.checkpoint)
        
        # 最终总结
        stats = evaluator.get_statistics()
        print("\n" + "="*70)
        print("✅ 批量评估完成！")
        print("="*70)
        print(f"📁 所有结果已保存至: {output_dir}")
        print(f"🎯 准确率: {stats['accuracy']*100:.2f}%")
        print(f"✅ 正确: {stats['correct']}/{stats['total']}")
        print(f"❌ 错误: {stats['wrong']}/{stats['total']}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

