import os
import json
import base64
import time
import logging
import numpy as np
import random
from PIL import Image
import io
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 种子值验证和修复函数
def validate_seed(seed, allow_random=True):
    """验证并修复seed值，确保在有效范围内
    
    Args:
        seed: 输入的seed值
        allow_random: 是否允许生成随机seed
    Returns:
        tuple: (修复后的seed值, 是否为随机生成的)
    """
    MAX_SEED = 2**31 - 1  # 2147483647 (32位有符号整数最大值)
    
    if seed == -1 and allow_random:
        # -1 表示随机生成种子
        new_seed = random.randint(0, MAX_SEED)
        logger.info(f"Seed为-1，生成随机种子: {new_seed}")
        return new_seed, True
    elif seed > MAX_SEED:
        # 超出范围，重新生成随机种子
        new_seed = random.randint(0, MAX_SEED)
        logger.warning(f"Seed值 {seed} 超出范围 [0, {MAX_SEED}]，重新生成随机种子: {new_seed}")
        return new_seed, True
    elif seed < 0:
        # 负值（非-1），修正为0
        logger.warning(f"Seed值 {seed} 为负值，修正为: 0")
        return 0, False
    else:
        # 正常范围内的值
        return seed, False

# 从文件中读取API Key

def get_api_key():
    api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ali_api_key.txt')
    try:
        with open(api_key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API key not found in ali_api_key.txt. Please add your API key to this file.")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"ali_api_key.txt not found at {api_key_path}. Please create this file and add your API key.")

# 图像编码为base64
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# 获取OpenAI客户端
def get_client():
    api_key = get_api_key()
    # 阿里云API endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return client

class Qwen3_VQA_API:
    """API版Qwen3_VQA节点"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 4096, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**63 - 1}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "prompt_text")

    def inference(self, prompt, temperature, max_new_tokens, top_p, frequency_penalty, presence_penalty, seed, **kwargs):
        try:
            start_time = time.time()
            logger.info(f"Starting Qwen3_VQA_API inference with prompt: {prompt[:50]}...")
            
            # 验证和修复seed值
            seed, is_random = validate_seed(seed)
            if is_random:
                logger.info(f"使用修复后的seed值: {seed}")
            
            # 获取客户端
            client = get_client()
            
            # 构建messages
            messages = []
            
            # 处理图像
            image1 = kwargs.get('image1')
            image2 = kwargs.get('image2')
            
            # 构建content
            content = []
            content.append({"type": "text", "text": prompt})
            
            # 添加图像1
            if image1 is not None:
                # Convert tensor to numpy array (handle both tensor and numpy cases)
                image_np = image1.cpu().numpy() if hasattr(image1, 'cpu') else image1
                # Scale and clip values to 0-255, convert to uint8
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                # Remove batch dimension if present (ComfyUI images usually have (batch, height, width, channel))
                if image_np.ndim == 4:
                    image_np = image_np[0]
                # Create PIL image
                pil_image = Image.fromarray(image_np)
                base64_image = image_to_base64(pil_image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            
            # 添加图像2
            if image2 is not None:
                # Convert tensor to numpy array (handle both tensor and numpy cases)
                image_np = image2.cpu().numpy() if hasattr(image2, 'cpu') else image2
                # Scale and clip values to 0-255, convert to uint8
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                # Remove batch dimension if present (ComfyUI images usually have (batch, height, width, channel))
                if image_np.ndim == 4:
                    image_np = image_np[0]
                # Create PIL image
                pil_image = Image.fromarray(image_np)
                base64_image = image_to_base64(pil_image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            
            messages.append({"role": "user", "content": content})
            
            # 打印API请求参数
            #logger.info(f"API请求参数: model='qwen-vl-plus', messages={messages}, temperature={temperature}, max_tokens={max_new_tokens}, top_p={top_p}, frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}, seed={seed}")
            # 调用API
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed
            )
            
            # 解析响应
            response_text = response.choices[0].message.content.strip()
            
            end_time = time.time()
            logger.info(f"Qwen3_VQA_API inference completed in {end_time - start_time:.2f} seconds")
            
            return (response_text, prompt)
            
        except Exception as e:
            logger.error(f"Qwen3_VQA_API inference error: {str(e)}", exc_info=True)
            return (f"Error: {str(e)}", "")

class Qwen3_VQA_Quick_API:
    """API版Qwen3_VQA_Quick节点"""
    def __init__(self):
        # 提示词模板文件夹路径
        self.prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        # 确保提示词文件夹存在
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir)

    @classmethod
    def INPUT_TYPES(s):
        # 获取提示词模板文件
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        if not os.path.exists(prompts_dir):
            os.makedirs(prompts_dir)
        
        # 加载支持的文件类型
        prompt_files = []
        if os.path.exists(prompts_dir):
            for file in os.listdir(prompts_dir):
                if file.endswith((".txt", ".md", ".json")):
                    prompt_files.append(file)
        
        return {
            "required": {
                "prompt_template": (sorted(prompt_files), {"default": prompt_files[0] if prompt_files else ""}),
                "user_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),  # 用户输入的辅助提示词
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 4096, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**63 - 1}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "prompt_text")

    def read_prompt_template(self, template_file):
        """读取提示词模板文件内容"""
        file_path = os.path.join(self.prompts_dir, template_file)
        if not os.path.exists(file_path):
            return ""  # 文件不存在返回空字符串
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 直接读取文件内容，不做特殊解析
                return f.read()
        except Exception as e:
            logger.error(f"读取提示词模板失败: {str(e)}")
            return ""  # 读取失败返回空字符串

    def inference(self, prompt_template, user_prompt, temperature, max_new_tokens, top_p, frequency_penalty, presence_penalty, seed, **kwargs):
        try:
            start_time = time.time()
            logger.info(f"Starting Qwen3_VQA_Quick_API inference...")
            
            # 验证和修复seed值
            seed, is_random = validate_seed(seed)
            if is_random:
                logger.info(f"使用修复后的seed值: {seed}")
            
            # 获取客户端
            client = get_client()
            
            # 读取选中的提示词模板
            template_content = self.read_prompt_template(prompt_template)
            
                # 构建messages
            messages = []
            
            # 处理图像
            image1 = kwargs.get('image1', None)
            image2 = kwargs.get('image2', None)
            
            # 添加系统提示（模板内容）
            if template_content:
                messages.append({"role": "system", "content": template_content})
            
            # 构建用户内容
            user_content = []
            
            # 添加用户提示
            if user_prompt:
                user_content.append({"type": "text", "text": user_prompt})
            
            # 添加图像1
            if image1 is not None:
                # Convert tensor to numpy array (handle both tensor and numpy cases)
                image_np = image1.cpu().numpy() if hasattr(image1, 'cpu') else image1
                # Scale and clip values to 0-255, convert to uint8
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                # Remove batch dimension if present (ComfyUI images usually have (batch, height, width, channel))
                if image_np.ndim == 4:
                    image_np = image_np[0]
                # Create PIL image
                pil_image = Image.fromarray(image_np)
                base64_image = image_to_base64(pil_image)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            
            # 添加图像2
            if image2 is not None:
                # Convert tensor to numpy array (handle both tensor and numpy cases)
                image_np = image2.cpu().numpy() if hasattr(image2, 'cpu') else image2
                # Scale and clip values to 0-255, convert to uint8
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                # Remove batch dimension if present (ComfyUI images usually have (batch, height, width, channel))
                if image_np.ndim == 4:
                    image_np = image_np[0]
                # Create PIL image
                pil_image = Image.fromarray(image_np)
                base64_image = image_to_base64(pil_image)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            
            # 只有当用户内容不为空时才添加用户消息
            if user_content:
                messages.append({"role": "user", "content": user_content})
            # 如果没有用户内容但有系统提示，则添加一个空的用户消息
            elif messages:  # 确保至少有一个用户消息来触发API
                messages.append({"role": "user", "content": [{"type": "text", "text": ""}]})
            # 打印API请求参数
            #logger.info(f"API请求参数: model='qwen-vl-plus', messages={messages}, temperature={temperature}, max_tokens={max_new_tokens}, top_p={top_p}, frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}, seed={seed}")
            
            # 调用API
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed
            )
            
            # 解析响应
            response_text = response.choices[0].message.content.strip()
            
            end_time = time.time()
            logger.info(f"Qwen3_VQA_Quick_API inference completed in {end_time - start_time:.2f} seconds")
            
            # 构建完整的提示文本以便返回
            combined_prompt = template_content + "\n" + user_prompt if template_content and user_prompt else template_content or user_prompt
            return (response_text, combined_prompt)
            
        except Exception as e:
            logger.error(f"Qwen3_VQA_Quick_API inference error: {str(e)}", exc_info=True)
            return (f"Error: {str(e)}", "")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA_API": Qwen3_VQA_API,
    "Qwen3_VQA_Quick_API": Qwen3_VQA_Quick_API,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA_API": "Qwen3 VQA API",
    "Qwen3_VQA_Quick_API": "Qwen3 VQA Quick API",
}