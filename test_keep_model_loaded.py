import sys
import os

# 将当前目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from nodes import Qwen3_VQA, Qwen3_VQA_Quick

def test_qwen3_vqa_keep_model_loaded():
    """测试Qwen3_VQA节点的keep_model_loaded选项"""
    print("=== 测试Qwen3_VQA节点的keep_model_loaded选项 ===")
    
    # 创建节点实例
    node = Qwen3_VQA()
    
    # 模拟参数
    model_id = "Qwen/Qwen-VL-Chat"
    quantization = "4bit"
    max_new_tokens = 100
    temperature = 0.1
    use_cache = False
    image = None  # 这里不使用图像
    image_path = None
    text = "Hello, how are you?"
    keep_model_loaded = True
    
    try:
        # 第一次调用（应该加载模型）
        print("\n--- 第一次调用（keep_model_loaded=True）---")
        result = node.inference(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            image=image,
            image_path=image_path,
            text=text,
            keep_model_loaded=keep_model_loaded
        )
        print(f"结果: {result[:100]}...")
        
        # 第二次调用（应该复用模型，不重新加载）
        print("\n--- 第二次调用（keep_model_loaded=True）---")
        result = node.inference(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            image=image,
            image_path=image_path,
            text=text,
            keep_model_loaded=keep_model_loaded
        )
        print(f"结果: {result[:100]}...")
        
        # 第三次调用（不保留模型，调用后应该释放）
        print("\n--- 第三次调用（keep_model_loaded=False）---")
        result = node.inference(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            image=image,
            image_path=image_path,
            text=text,
            keep_model_loaded=False
        )
        print(f"结果: {result[:100]}...")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_qwen3_vqa_quick_keep_model_loaded():
    """测试Qwen3_VQA_Quick节点的keep_model_loaded选项"""
    print("\n=== 测试Qwen3_VQA_Quick节点的keep_model_loaded选项 ===")
    
    # 创建节点实例
    node = Qwen3_VQA_Quick()
    
    # 模拟参数
    model_id = "Qwen/Qwen-VL-Chat"
    quantization = "4bit"
    max_new_tokens = 100
    temperature = 0.1
    use_cache = False
    image = None  # 这里不使用图像
    image_path = None
    text = "Hello, how are you?"
    keep_model_loaded = True
    prompt_template = "{{text}}"
    
    try:
        # 第一次调用（应该加载模型）
        print("\n--- 第一次调用（keep_model_loaded=True）---")
        result = node.inference(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            image=image,
            image_path=image_path,
            text=text,
            keep_model_loaded=keep_model_loaded,
            prompt_template=prompt_template
        )
        print(f"结果: {result[:100]}...")
        
        # 第二次调用（应该复用模型，不重新加载）
        print("\n--- 第二次调用（keep_model_loaded=True）---")
        result = node.inference(
            model_id=model_id,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            image=image,
            image_path=image_path,
            text=text,
            keep_model_loaded=keep_model_loaded,
            prompt_template=prompt_template
        )
        print(f"结果: {result[:100]}...")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    
    # 测试两个节点
    test_qwen3_vqa_keep_model_loaded()
    test_qwen3_vqa_quick_keep_model_loaded()