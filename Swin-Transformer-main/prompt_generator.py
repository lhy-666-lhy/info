#!/usr/bin/env python3
"""
Prompt生成器
输入图片路径，自动进行推理并生成包含图片和分析的Markdown报告
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
import cv2
from datetime import datetime
from pathlib import Path
import shutil

# 使用脚本所在目录为项目根，便于任意路径下运行
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from config import get_config
from models import build_model


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"📥 加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    class_names = checkpoint.get('class_names', ['Class 0', 'Class 1', 'Class 2'])
    
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
    
    os.environ['LOCAL_RANK'] = '0'
    args = Args()
    config = get_config(args)
    
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()
    
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    return model, class_names


def preprocess_image(image_path, img_size=224):
    """预处理图片"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize((img_size, img_size)))
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, original_image


def predict_image(model, image_tensor):
    """预测图片类别"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def generate_gradcam(model, image_tensor, original_image, predicted_class):
    """生成Grad-CAM热力图"""
    # 找到最后一个norm层
    target_layer = None
    for name, module in model.named_modules():
        if 'norm' in name and isinstance(module, torch.nn.LayerNorm):
            target_layer = module
    
    if target_layer is None:
        return None
    
    try:
        gradients = None
        activations = None
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output
        
        def save_gradient(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        # 注册hooks
        h1 = target_layer.register_forward_hook(save_activation)
        h2 = target_layer.register_full_backward_hook(save_gradient)
        
        # 前向传播
        model.train()
        image_tensor.requires_grad = True
        model_output = model(image_tensor)
        
        # 反向传播
        model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0][predicted_class] = 1
        model_output.backward(gradient=one_hot)
        
        # 移除hooks
        h1.remove()
        h2.remove()
        
        # 计算CAM
        grads = gradients.detach().cpu()
        acts = activations.detach().cpu()
        
        # 处理形状
        if len(acts.shape) == 3:  # [B, H*W, C]
            B, N, C = acts.shape
            H = W = int(np.sqrt(N))
            acts = acts.transpose(1, 2).reshape(B, C, H, W)
            grads = grads.transpose(1, 2).reshape(B, C, H, W)
        
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        model.eval()
        
        # 调整到原始图像大小
        cam_resized = cv2.resize(cam.squeeze().numpy(), 
                                (original_image.shape[1], original_image.shape[0]))
        
        return cam_resized
        
    except Exception as e:
        print(f"  ⚠️  Grad-CAM生成失败: {e}")
        model.eval()
        return None


def generate_attention_overlay(model, image_tensor, original_image):
    """生成注意力可视化叠加图"""
    try:
        model.eval()
        image_tensor.requires_grad = True
        
        # 前向传播
        output = model(image_tensor)
        pred_class = output.argmax(dim=1)
        
        # 反向传播获取梯度注意力
        model.zero_grad()
        output[0, pred_class].backward()
        
        # 获取梯度
        gradients = image_tensor.grad.data.abs()
        
        # 计算每个通道的平均梯度作为注意力
        attention = gradients[0].mean(0).cpu().numpy()
        
        # 归一化
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # 调整大小
        attention_resized = cv2.resize(attention, (original_image.shape[1], original_image.shape[0]))
        
        # 转换为彩色热力图
        attention_colored = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_HOT)
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加到原图
        overlay = attention_colored * 0.5 + original_image * 0.5
        overlay = np.uint8(overlay)
        
        return overlay
        
    except Exception as e:
        print(f"  ⚠️  注意力可视化生成失败: {e}")
        return None


def create_visualization_images(original_image, cam, attention_overlay, output_dir):
    """创建三张可视化图片：原图、Grad-CAM叠加、注意力叠加"""
    
    # 图片1: 原始图像
    img1_path = os.path.join(output_dir, '1_original.png')
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('原始图像', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(img1_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图片2: Grad-CAM叠加效果
    img2_path = os.path.join(output_dir, '2_gradcam_overlay.png')
    if cam is not None:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = heatmap * 0.4 + original_image * 0.6
        superimposed = np.uint8(superimposed)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(superimposed)
        plt.axis('off')
        plt.title('Grad-CAM 叠加效果', fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(img2_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # 如果CAM生成失败，使用原图
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image)
        plt.axis('off')
        plt.title('Grad-CAM叠加效果（生成失败）', fontsize=14, pad=10)
        plt.savefig(img2_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 图片3: 注意力可视化叠加效果
    img3_path = os.path.join(output_dir, '3_attention_overlay.png')
    if attention_overlay is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(attention_overlay)
        plt.axis('off')
        plt.title('注意力可视化叠加效果', fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(img3_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # 如果注意力生成失败，使用原图
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image)
        plt.axis('off')
        plt.title('注意力叠加效果（生成失败）', fontsize=14, pad=10)
        plt.savefig(img3_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return img1_path, img2_path, img3_path





def main():
    parser = argparse.ArgumentParser('Prompt生成器 - 细胞图像AI分析')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--checkpoint', type=str, 
                       default='output/exp_combined/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--output-base', type=str, default='evaluation_results',
                       help='输出基础目录')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🤖 细胞图像AI分析系统 - Prompt生成器")
    print("="*70)
    print(f"🖼️  输入图片: {args.image}")
    print(f"💾 AI模型: {args.checkpoint}")
    print("="*70 + "\n")
    
    # 检查文件
    if not os.path.exists(args.image):
        print(f"❌ 图片文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ 模型文件不存在: {args.checkpoint}")
        return
    
    try:
        # 创建输出目录
        image_name = Path(args.image).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_base, f"{image_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"OUTPUT_DIR:{output_dir}")  # 立即输出供 shell 解析，避免后续异常时无法获取
        print(f"📂 创建输出目录: {output_dir}\n")
        
        # 加载模型
        model, class_names = load_model(args.checkpoint)
        
        # 预处理图片
        print("🔄 预处理图片...")
        image_tensor, original_image = preprocess_image(args.image)
        print("✅ 预处理完成\n")
        
        # AI预测
        print("🔍 AI分析中...")
        predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)
        
        # 输出预测结果（机器可读格式）
        print("\n" + "="*70)
        print("AI_RESULT_START")
        print(f"PREDICTED_CLASS:{class_names[predicted_class]}")
        print(f"CONFIDENCE:{confidence:.4f}")
        for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
            print(f"PROB_{class_name}:{prob:.4f}")
        print("AI_RESULT_END")
        print("="*70 + "\n")
        
        # 生成Grad-CAM
        print("🎨 生成Grad-CAM可视化...")
        cam = generate_gradcam(model, image_tensor.clone(), original_image, predicted_class)
        
        # 生成注意力可视化
        print("👁️  生成注意力可视化...")
        attention_overlay = generate_attention_overlay(model, image_tensor.clone(), original_image)
        
        # 创建三张图片
        print("📸 生成可视化图片...")
        img1_path, img2_path, img3_path = create_visualization_images(
            original_image, cam, attention_overlay, output_dir
        )
        print(f"  ✅ 图片1 (原图): {os.path.basename(img1_path)}")
        print(f"  ✅ 图片2 (Grad-CAM叠加): {os.path.basename(img2_path)}")
        print(f"  ✅ 图片3 (注意力叠加): {os.path.basename(img3_path)}")
        
        
        # 最终总结
        print("\n" + "="*70)
        print("OUTPUT_DIR:" + output_dir)
        print("IMG1:" + img1_path)
        print("IMG2:" + img2_path)
        print("IMG3:" + img3_path)
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

