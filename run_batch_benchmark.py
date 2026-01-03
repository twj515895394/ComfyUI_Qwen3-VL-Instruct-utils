import sys
import os
import time
import torch
import numpy as np
from PIL import Image

# 1. 设置环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
sys.path.append(comfy_root)

# 模拟 ComfyUI 路径
import folder_paths
folder_paths.models_dir = os.path.join(comfy_root, "models")
folder_paths.temp_directory = os.path.join(comfy_root, "temp")

# 导入节点
from nodes import Qwen3_VQA_Quick, Qwen3_VQA_Batch, Qwen3ModelManager

def load_images_batch(test_pic_dir):
    """加载目录下所有 jpg 图片并堆叠为 Batch Tensor [B, H, W, C]"""
    img_list = []
    files = sorted([f for f in os.listdir(test_pic_dir) if f.endswith(".jpg")])
    if not files:
        print("No jpg images found in test_pic")
        return None
    
    print(f"Loading {len(files)} images for batch test...")
    for f in files:
        img_path = os.path.join(test_pic_dir, f)
        img = Image.open(img_path).convert("RGB")
        # 为简单起见，Resize 到相同大小以方便 torch.stack (ComfyUI 通常会自动处理 batch 大小一致性，或者这里模拟 stack)
        # 注意：ComfyUI 的 Load Image Batch 节点通常要求尺寸一致。
        # 如果尺寸不一致，nodes.py 里的 _tensor_to_pil 会逐个处理，所以这里的输入可以是 list of tensors，
        # 但 Qwen3_VQA_Batch 的 INPUT_TYPES 声明的是 IMAGE (Tensor)。
        # 为了测试方便，我们这里假设输入已经是 Stack 好的 Tensor。
        img = img.resize((512, 512)) 
        img_np = np.array(img).astype(np.float32) / 255.0
        img_list.append(torch.from_numpy(img_np))
    
    batch_tensor = torch.stack(img_list) # [B, H, W, C]
    return batch_tensor

def run_test():
    print("=== 开始批量性能对比测试 ===")
    
    test_pic_dir = os.path.join(current_dir, "test_pic")
    batch_images = load_images_batch(test_pic_dir)
    if batch_images is None: return

    # 公共参数
    common_params = {
        "prompt_template": "基础反推图片生成提示词.txt",
        "model": "Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "keep_model_loaded": True,
        "temperature": 0.0, # Greedy for stable benchmark
        "max_new_tokens": 256,
        "min_pixels": 256*28*28,
        "max_pixels": 768*28*28,
        "seed": 42,
        "quantization": "none",
        "attention": "sage_attention_2",
        "performance_mode": "speed"
    }

    # 预热模型 (Pre-warm)
    print("\n[Pre-warm] 加载模型中...")
    warmup_node = Qwen3_VQA_Quick()
    # 随便跑一张
    warmup_node.run_inference(image1=batch_images[0:1], user_prompt="", source_path=None, image2=None, **common_params)
    print("模型已加载。")

    # --- 测试 1: Loop Mode (模拟 ComfyUI 外部循环) ---
    print("\n[测试 1] Loop Mode (单图循环 x10)...")
    single_node = Qwen3_VQA_Quick()
    
    start_t = time.time()
    for i in range(batch_images.shape[0]):
        # 取单张 [1, H, W, C]
        single_img = batch_images[i:i+1]
        # 为了更真实，这里不清除 cache，因为每张图内容不同，cache key 也不同
        single_node.run_inference(image1=single_img, user_prompt="", source_path=None, image2=None, **common_params)
        print(f"  Image {i+1} done.")
    
    loop_duration = time.time() - start_t
    print(f"Loop Mode 总耗时: {loop_duration:.2f}s | 平均: {loop_duration/batch_images.shape[0]:.2f}s/img")

    # --- 测试 2: Batch Mode (Qwen3_VQA_Batch) ---
    print("\n[测试 2] Batch Mode (Batch Node x1)...")
    batch_node = Qwen3_VQA_Batch()
    
    # 尝试 Batch Size = 4 (显存允许的话)
    bs = 4
    print(f"Setting Batch Size = {bs}")
    
    start_t = time.time()
    result = batch_node.run_inference_batch(
        images=batch_images, 
        batch_size=bs,
        user_prompt="",
        **common_params
    )
    
    batch_duration = time.time() - start_t
    print(f"Batch Mode 总耗时: {batch_duration:.2f}s | 平均: {batch_duration/batch_images.shape[0]:.2f}s/img")
    
    # --- 结果对比 ---
    speedup = loop_duration / batch_duration
    print(f"\n=== 结论 ===")
    print(f"加速比: {speedup:.2f}x")
    if speedup > 1.0:
        print("Batch Mode 显著更快！")
    else:
        print("Batch Mode 没有变快，可能是显存带宽瓶颈或 Padding 开销过大。")

    Qwen3ModelManager.get_instance().unload_all()

if __name__ == "__main__":
    run_test()
