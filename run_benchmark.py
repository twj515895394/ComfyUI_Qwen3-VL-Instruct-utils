import sys
import os
import time
import torch
import numpy as np
from PIL import Image

# 1. 设置环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # Assuming X:\ComfyUI-aki-v2\ComfyUI
sys.path.append(comfy_root)

# 模拟 ComfyUI 的 folder_paths
import folder_paths
folder_paths.models_dir = os.path.join(comfy_root, "models")
folder_paths.temp_directory = os.path.join(comfy_root, "temp")

# 导入节点
from nodes import Qwen3_VQA_Quick, Qwen3ModelManager

def load_image_as_tensor(image_path):
    """加载图片并转换为 ComfyUI 格式的 Tensor [1, H, W, C]"""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None, ...]
    return img_tensor

def run_test():
    print("=== 开始性能测试 ===")
    
    # 准备测试数据
    test_img_path = os.path.join(current_dir, "test_pic", "test.jpg")
    prompt_template = "基础反推图片生成提示词.txt"
    
    if not os.path.exists(test_img_path):
        print(f"Error: 测试图片不存在: {test_img_path}")
        return

    print(f"加载测试图片: {test_img_path}")
    image_tensor = load_image_as_tensor(test_img_path)
    
    # 实例化节点
    node = Qwen3_VQA_Quick()
    
    # 参数配置 - 最终优化版 (目标 < 20s)
    params = {
        "prompt_template": prompt_template,
        "model": "Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "keep_model_loaded": True,
        "temperature": 0, 
        "max_new_tokens": 256, # 256 tokens 足够大多数 caption 需求
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 768 * 28 * 28, # 768 是性能与质量的平衡点
        "seed": 42,
        "quantization": "none", 
        "user_prompt": "",
        "source_path": None,
        "image1": image_tensor,
        "image2": None,
        "attention": "sage_attention_2",
        "performance_mode": "speed"
    }

    # --- 冷启动测试 ---
    print("\n[测试 1] 冷启动 (含模型加载)...")
    start_time = time.time()
    try:
        result1 = node.run_inference(**params)
        end_time = time.time()
        print(f"冷启动总耗时: {end_time - start_time:.2f} 秒")
        print(f"输出结果预览: {result1[0][:100]}...")
    except Exception as e:
        print(f"冷启动失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 热启动测试 ---
    print("\n[测试 2] 热启动 (模型已加载)...")
    
    # 稍微修改参数以避开缓存 (虽然我们没改 seed，但为了确保走 inference 逻辑，可以手动清空 cache 或依赖 key 生成)
    # 这里直接调用，因为 seed 固定，如果 cache 命中会瞬间返回。
    # 为了测试推理性能，我们需要让 cache 失效。
    node.cache.clear() 
    
    start_time = time.time()
    try:
        result2 = node.run_inference(**params)
        end_time = time.time()
        print(f"热启动总耗时: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"热启动失败: {e}")

    # 清理
    Qwen3ModelManager.get_instance().unload_all()
    print("\n测试结束。\n")

if __name__ == "__main__":
    run_test()
