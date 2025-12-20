import os
import torch
import time
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import comfy.model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path
import json


class ModelManager:
    """模型管理器，使用智能缓存管理模型生命周期"""
    _model = None
    _processor = None
    _current_model_id = None
    _current_quantization = None
    _reference_count = 0
    _active_sessions = set()  # 跟踪活跃的节点实例
    _last_model_id = None  # 保留最后使用的模型ID
    _last_quantization = None  # 保留最后使用的量化方式
    
    @classmethod
    def acquire_model(cls, session_id):
        """获取模型，增加引用计数"""
        cls._active_sessions.add(session_id)
        cls._reference_count += 1
        print(f"[ModelManager] 模型引用计数: {cls._reference_count}, 活跃会话: {len(cls._active_sessions)}")
        
    @classmethod
    def release_model(cls, session_id):
        """释放模型，减少引用计数"""
        if session_id in cls._active_sessions:
            cls._active_sessions.remove(session_id)
            cls._reference_count = max(0, cls._reference_count - 1)
            print(f"[ModelManager] 模型引用计数: {cls._reference_count}, 活跃会话: {len(cls._active_sessions)}")
            
            # 修改逻辑：只有在引用计数为0且模型真的需要释放时才释放
            # 或者在特定条件下延迟释放模型
            if cls._reference_count == 0:
                # 检查是否应该延迟释放模型
                if cls._should_keep_model_loaded():
                    print(f"[ModelManager] 启用keep_model_loaded，延迟释放模型")
                    return
                cls._release_all_resources()
    
    @classmethod
    def _should_keep_model_loaded(cls):
        """判断是否应该保持模型加载状态"""
        # 如果模型已经加载且不是None，说明模型可用
        return cls._model is not None and cls._processor is not None
    
    @classmethod
    def _release_all_resources(cls):
        """释放所有模型资源"""
        print(f"[ModelManager] 释放所有模型资源")
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._processor is not None:
            del cls._processor
            cls._processor = None
        # 保留模型ID和量化信息，避免重复加载判断错误
        # cls._current_model_id = None
        # cls._current_quantization = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    @classmethod
    def get_model_info(cls):
        """获取当前模型信息"""
        return {
            'model': cls._model,
            'processor': cls._processor,
            'current_model_id': cls._current_model_id,
            'current_quantization': cls._current_quantization,
            'reference_count': cls._reference_count,
            'active_sessions': len(cls._active_sessions)
        }

class Qwen3_Base:
    """Qwen3节点的基类，包含通用的缓存功能和模型管理"""
    
    def __init__(self):
        self.model_checkpoint = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.cache = {}  # 用于存储输入参数和模型输出的缓存
        self.max_cache_size = 100  # 最大缓存条目数
        self.cache_enabled = True  # 缓存开关
        self.session_id = id(self)  # 唯一会话标识符
        print(f"[{self.__class__.__name__}] 节点初始化完成，会话ID: {self.session_id}")
    
    def _create_cache_key(self, **kwargs):
        """创建缓存键，基于所有输入参数"""
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if v is not None:
                if isinstance(v, torch.Tensor):
                    # 对于图像张量，增强缓存键的生成逻辑，添加更多特征信息
                    try:
                        # 使用形状、均值、标准差和张量数据的哈希值
                        # 对于大张量，采样部分数据进行哈希，避免计算开销过大
                        shape_str = str(v.shape)
                        mean_val = v.mean().item()
                        std_val = v.std().item()
                        
                        # 采样部分数据计算哈希值（如果张量很大）
                        if v.numel() > 10000:  # 如果张量元素数量超过10000
                            # 均匀采样100个点
                            indices = torch.linspace(0, v.numel() - 1, min(100, v.numel()), dtype=torch.long)
                            sampled_data = v.view(-1)[indices]
                            hash_val = hash(str(sampled_data.cpu().numpy().tolist()))
                        else:
                            # 对于小张量，使用所有数据的哈希值
                            hash_val = hash(str(v.cpu().numpy().tolist()))
                        
                        key_parts.append(f"{k}:{shape_str}:{mean_val:.4f}:{std_val:.4f}:{hash_val}")
                    except Exception as e:
                        # 如果出现异常，降级使用基本信息
                        print(f"[缓存键生成警告] 处理张量 {k} 时出错: {e}，使用基本信息")
                        key_parts.append(f"{k}:{v.shape}:{v.mean().item():.4f}:{v.std().item():.4f}")
                else:
                    key_parts.append(f"{k}:{v}")
        
        # 为了确保缓存键的唯一性和稳定性，添加整体哈希
        cache_key = "_".join(key_parts)
        # 如果键太长，使用哈希值缩短
        if len(cache_key) > 1000:
            cache_key = f"hash:{hash(cache_key)}"
            
        return cache_key
    
    def generate_cache_key_without_random_seed(self, seed, **kwargs):
        """生成缓存键，当seed=-1时不包含在缓存键中"""
        cache_params = dict(kwargs)
        # 只有当seed不是-1时才将其包含在缓存键中
        if seed != -1:
            cache_params['seed'] = seed
        
        return self._create_cache_key(**cache_params)
    
    def check_cache(self, cache_key, use_cache=True):
        """检查缓存是否存在并返回结果"""
        if use_cache and self.cache_enabled and cache_key in self.cache:
            print(f"[{self.__class__.__name__}] 缓存命中! 使用缓存结果。")
            return self.cache[cache_key]
        if use_cache and self.cache_enabled:
            print(f"[{self.__class__.__name__}] 缓存未命中，将请求模型。")
        return None
    
    def update_cache(self, cache_key, result, use_cache=True):
        """更新缓存"""
        if use_cache and self.cache_enabled:
            self.cache[cache_key] = result
            print(f"[{self.__class__.__name__}] 结果已缓存。当前缓存大小: {len(self.cache)}/{self.max_cache_size}")
            # 限制缓存大小
            if len(self.cache) > self.max_cache_size:
                # 删除最早添加的项目
                self.cache.pop(next(iter(self.cache)))
                print(f"[{self.__class__.__name__}] 缓存已满，已删除最早的缓存项。")


class Qwen3_VQA(Qwen3_Base):
    def __init__(self):
        super().__init__()  # 调用基类的初始化方法
        print("[Qwen3_VQA] 节点初始化完成，缓存系统已设置")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                        "Huihui-Qwen3-VL-8B-Instruct-abliterated",
                    ],
                    {"default": "Huihui-Qwen3-VL-8B-Instruct-abliterated"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                    {"default": "flash_attention_2"},  # 默认使用最高效的注意力机制
                ),
                "use_cache": ("BOOLEAN", {"default": True}),  # 缓存开关
                "performance_mode": (
                    [
                        "balanced",  # 平衡模式
                        "speed",     # 速度优先
                        "quality",   # 质量优先
                    ],
                    {"default": "balanced"},  # 默认平衡模式
                ),
                "max_batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1},
                    {"forceInput": False}
                ),  # 批处理大小限制
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def inference(
            self,
            text,
            model,
            keep_model_loaded,
            temperature,
            max_new_tokens,
            min_pixels,
            max_pixels,
            seed,
            quantization,
            source_path=None,
            image=None,  # add image parameter
            attention="flash_attention_2",
            use_cache=True,
            performance_mode="speed",
            max_batch_size=1,
    ):
        # 使用基类的缓存键生成方法
        cache_key = self.generate_cache_key_without_random_seed(
            seed=seed,
            text=text,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            quantization=quantization,
            source_path=source_path,
            image=image,
            attention=attention,
            performance_mode=performance_mode,
            max_batch_size=max_batch_size
        )
        
        # 检查是否可以从缓存获取结果
        cached_result = self.check_cache(cache_key, use_cache)
        if cached_result is not None:
            return (cached_result,)
            
        # 增加模型引用计数
        ModelManager.acquire_model(self.session_id)
        
        try:
            if seed != -1:
                torch.manual_seed(seed)
            if model == "Huihui-Qwen3-VL-8B-Instruct-abliterated":
                model_id = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
            else:
                model_id = f"qwen/{model}"
            self.model_checkpoint = f"Y:\\llama-models\\Qwen3-vl-nsfw\\prompt_generator\\Huihui-Qwen3-VL-8B-Instruct-abliterated"

            if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model_id,
                    local_dir=self.model_checkpoint,
                    allow_patterns=["*.json", "*.bin", "*.model", "*.pth"],
                    force_download=False,
                )

            # 获取当前模型状态
            model_info = ModelManager.get_model_info()
            print(f"[{self.__class__.__name__}] 当前模型ID: {model_info['current_model_id']}, 目标模型ID: {model_id}")
            print(f"[{self.__class__.__name__}] 当前量化方式: {model_info['current_quantization']}, 目标量化方式: {quantization}")
            print(f"[{self.__class__.__name__}] 当前processor状态: {'已加载' if model_info['processor'] is not None else '未加载'}")
            print(f"[{self.__class__.__name__}] 当前model状态: {'已加载' if model_info['model'] is not None else '未加载'}")
            
            if (
                    model_info['current_model_id'] != model_id
                    or model_info['current_quantization'] != quantization
                    or model_info['processor'] is None
                    or model_info['model'] is None
            ):
                print(f"[{self.__class__.__name__}] 模型或处理器需要重新加载")
                ModelManager._current_model_id = model_id
                ModelManager._current_quantization = quantization
                if model_info['processor'] is not None:
                    del model_info['processor']
                if model_info['model'] is not None:
                    del model_info['model']
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                # 加载处理器
                start_time = time.time()
                ModelManager._processor = AutoProcessor.from_pretrained(
                        self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
                    )
                processor_load_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 处理器加载完成，耗时: {processor_load_time:.2f}秒")
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                    )
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                else:
                    quantization_config = None

                # 加载模型
                start_time = time.time()
                ModelManager._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    attn_implementation=attention,
                    quantization_config=quantization_config,
                )
                model_load_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 模型加载完成，耗时: {model_load_time:.2f}秒")
                print(f"[{self.__class__.__name__}] 模型和处理器加载完成，总耗时: {processor_load_time + model_load_time:.2f}秒")
            else:
                print(f"[{self.__class__.__name__}] 复用现有模型和处理器")

            temp_path = None
            if image is not None:
                pil_image = ToPILImage()(image[0].permute(2, 0, 1))
                temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
                pil_image.save(temp_path)

            with torch.no_grad():
                if source_path:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                        },
                        {
                            "role": "user",
                            "content": source_path
                                       + [
                                           {"type": "text", "text": text},
                                       ],
                        },
                    ]
                elif temp_path:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{temp_path}"},
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                            ],
                        }
                    ]

                # Preparation for inference
                text = ModelManager._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = ModelManager._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                # Inference: Generation of the output
                start_time = time.time()
                
                # 根据性能模式智能调整推理参数
                if performance_mode == "speed":
                    # 速度优先模式：减少推理步骤，优化性能
                    if temperature == 0:
                        # 温度为0时使用贪婪解码，最快
                        generation_config = {
                            "max_new_tokens": min(max_new_tokens, 1024),  # 限制生成长度
                            "temperature": 0.0,
                            "do_sample": False,
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                    else:
                        # 温度大于0时使用轻量级采样
                        generation_config = {
                            "max_new_tokens": min(max_new_tokens, 1024),  # 限制生成长度
                            "temperature": temperature,
                            "do_sample": True,
                            "top_p": 0.8,  # 降低top_p以减少计算
                            "top_k": 30,   # 降低top_k以减少计算
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                elif performance_mode == "quality":
                    # 质量优先模式：最大化推理质量
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "do_sample": True,
                        "top_p": 0.95,  # 提高top_p以获得更多样化的输出
                        "top_k": 100,   # 增加top_k以获得更多候选
                        "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                    }
                else:  # balanced模式
                    # 平衡模式：根据温度智能选择解码策略
                    if temperature == 0:
                        # 温度为0时使用贪婪解码，更快且确定性
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.0,
                            "do_sample": False,
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                    else:
                        # 温度大于0时使用采样解码
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "do_sample": True,
                            "top_p": 0.9,  # 添加核采样，提高质量
                            "top_k": 50,   # 限制候选词数量
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                
                generated_ids = ModelManager._model.generate(
                    **inputs,
                    **generation_config
                )
                inference_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 推理完成，耗时: {inference_time:.2f}秒")
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = ModelManager._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )

                # 将结果存入缓存
                self.update_cache(cache_key, result, use_cache)
                
                print(f"[Qwen3_VQA] 推理完成")
                return (result,)
                
        finally:
            # 减少模型引用计数
            ModelManager.release_model(self.session_id)
    
    


class Qwen3_VQA_Quick(Qwen3_Base):
    def __init__(self):
        super().__init__()  # 调用基类的初始化方法
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
                "model": (
                    [
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                        "Huihui-Qwen3-VL-8B-Instruct-abliterated",
                    ],
                    {"default": "Huihui-Qwen3-VL-8B-Instruct-abliterated"},
                ),
                "user_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),  # 用户输入的辅助提示词
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                    {"default": "flash_attention_2"},  # 默认使用最高效的注意力机制
                ),
                "use_cache": ("BOOLEAN", {"default": True}),  # 缓存开关
                "performance_mode": (
                    [
                        "balanced",  # 平衡模式
                        "speed",     # 速度优先
                        "quality",   # 质量优先
                    ],
                    {"default": "balanced"},  # 默认平衡模式
                ),
                "max_batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1},
                    {"forceInput": False}
                ),  # 批处理大小限制
            },
            "optional": {
                "source_path": ("PATH",),
                "image1": ("IMAGE",),  # 第一个图像输入（首帧）
                "image2": ("IMAGE",),  # 第二个图像输入（尾帧）
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # 增加一个输出，返回实际使用的提示词
    RETURN_NAMES = ("response", "prompt_text")
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

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
            print(f"读取提示词模板失败: {e}")
            return ""  # 读取失败返回空字符串

    def _create_cache_key(self, **kwargs):
        """创建缓存键，基于所有输入参数"""
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if v is not None:
                if isinstance(v, torch.Tensor):
                    # 对于图像张量，增强缓存键的生成逻辑，添加更多特征信息
                    try:
                        # 使用形状、均值、标准差和张量数据的哈希值
                        # 对于大张量，采样部分数据进行哈希，避免计算开销过大
                        shape_str = str(v.shape)
                        mean_val = v.mean().item()
                        std_val = v.std().item()
                        
                        # 采样部分数据计算哈希值（如果张量很大）
                        if v.numel() > 10000:  # 如果张量元素数量超过10000
                            # 均匀采样100个点
                            indices = torch.linspace(0, v.numel() - 1, min(100, v.numel()), dtype=torch.long)
                            sampled_data = v.view(-1)[indices]
                            hash_val = hash(str(sampled_data.cpu().numpy().tolist()))
                        else:
                            # 对于小张量，使用所有数据的哈希值
                            hash_val = hash(str(v.cpu().numpy().tolist()))
                        
                        key_parts.append(f"{k}:{shape_str}:{mean_val:.4f}:{std_val:.4f}:{hash_val}")
                    except Exception as e:
                        # 如果出现异常，降级使用基本信息
                        print(f"[缓存键生成警告] 处理张量 {k} 时出错: {e}，使用基本信息")
                        key_parts.append(f"{k}:{v.shape}:{v.mean().item():.4f}:{v.std().item():.4f}")
                else:
                    key_parts.append(f"{k}:{v}")
        
        # 为了确保缓存键的唯一性和稳定性，添加整体哈希
        cache_key = "_".join(key_parts)
        # 如果键太长，使用哈希值缩短
        if len(cache_key) > 1000:
            cache_key = f"hash:{hash(cache_key)}"
            
        return cache_key

    def inference(
            self,
            prompt_template,
            model,
            keep_model_loaded,
            temperature,
            max_new_tokens,
            min_pixels,
            max_pixels,
            seed,
            quantization,
            user_prompt="",
            source_path=None,
            image1=None,
            image2=None,
            attention="flash_attention_2",
            use_cache=True,
            performance_mode="speed",
            max_batch_size=1,
    ):
        # 使用基类的缓存键生成方法
        cache_key = self.generate_cache_key_without_random_seed(
            seed=seed,
            prompt_template=prompt_template,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            quantization=quantization,
            source_path=source_path,
            image1=image1,
            image2=image2,
            attention=attention,
            performance_mode=performance_mode,
            max_batch_size=max_batch_size
        )
        
        # 检查是否可以从缓存获取结果
        cached_result = self.check_cache(cache_key, use_cache)
        if cached_result is not None:
            result, text = cached_result
            print(f"[Qwen3_VQA_Quick] 推理完成")
            return (result, text)
        
        # 读取提示词模板内容
        template_text = self.read_prompt_template(prompt_template)
        # 拼接用户输入的辅助提示词
        text = template_text
        if user_prompt.strip():
            text += f"\n\nUser prompt:{user_prompt}"
        
        # 增加模型引用计数
        ModelManager.acquire_model(self.session_id)
        
        try:
            if seed != -1:
                torch.manual_seed(seed)
            if model == "Huihui-Qwen3-VL-8B-Instruct-abliterated":
                model_id = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
            else:
                model_id = f"qwen/{model}"
            
            # 这里使用硬编码路径，与原节点保持一致
            if model == "Huihui-Qwen3-VL-8B-Instruct-abliterated":
                self.model_checkpoint = f"Y:\\llama-models\\Qwen3-vl-nsfw\\prompt_generator\\Huihui-Qwen3-VL-8B-Instruct-abliterated"
            else:
                self.model_checkpoint = os.path.join(
                    folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
                )

            if not os.path.exists(self.model_checkpoint):
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model_id,
                    local_dir=self.model_checkpoint,
                    allow_patterns=["*.json", "*.bin", "*.model", "*.pth"],
                    force_download=False,
                )

            # 获取当前模型状态
            model_info = ModelManager.get_model_info()
            print(f"[{self.__class__.__name__}] 当前模型ID: {model_info['current_model_id']}, 目标模型ID: {model_id}")
            print(f"[{self.__class__.__name__}] 当前量化方式: {model_info['current_quantization']}, 目标量化方式: {quantization}")
            print(f"[{self.__class__.__name__}] 当前processor状态: {'已加载' if model_info['processor'] is not None else '未加载'}")
            print(f"[{self.__class__.__name__}] 当前model状态: {'已加载' if model_info['model'] is not None else '未加载'}")
            
            # 智能模型复用逻辑：只有当需要重新加载模型时才释放
            need_reload = (
                model_info['current_model_id'] != model_id
                or model_info['current_quantization'] != quantization
                or model_info['processor'] is None
                or model_info['model'] is None
            )
            
            if keep_model_loaded and model_info['model'] is not None and model_info['processor'] is not None:
                # 如果keep_model_loaded开启且模型已存在，检查是否匹配
                if (model_info['current_model_id'] == model_id and 
                    model_info['current_quantization'] == quantization):
                    print(f"[{self.__class__.__name__}] keep_model_loaded启用，复用现有模型和处理器")
                    # 更新当前模型信息以保持一致性
                    ModelManager._current_model_id = model_id
                    ModelManager._current_quantization = quantization
                else:
                    print(f"[{self.__class__.__name__}] keep_model_loaded启用但模型ID不匹配，需要重新加载")
                    need_reload = True
            elif need_reload:
                print(f"[{self.__class__.__name__}] 模型或处理器需要重新加载")
                # 清理之前的模型资源
                if model_info['processor'] is not None:
                    del model_info['processor']
                if model_info['model'] is not None:
                    del model_info['model']
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            if need_reload:
                ModelManager._current_model_id = model_id
                ModelManager._current_quantization = quantization
                # 加载处理器
                start_time = time.time()
                ModelManager._processor = AutoProcessor.from_pretrained(
                    self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
                )
                processor_load_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 处理器加载完成，耗时: {processor_load_time:.2f}秒")
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                    )
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                else:
                    quantization_config = None

                # 加载模型
                start_time = time.time()
                ModelManager._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    attn_implementation=attention,
                    quantization_config=quantization_config,
                )
                model_load_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 模型加载完成，耗时: {model_load_time:.2f}秒")
                print(f"[{self.__class__.__name__}] 模型和处理器加载完成，总耗时: {processor_load_time + model_load_time:.2f}秒")
            else:
                print(f"[{self.__class__.__name__}] 复用现有模型和处理器")

            temp_paths = []
            if image1 is not None:
                pil_image1 = ToPILImage()(image1[0].permute(2, 0, 1))
                temp_path1 = Path(folder_paths.temp_directory) / f"temp_image1_{seed}.png"
                pil_image1.save(temp_path1)
                temp_paths.append(temp_path1)
            
            if image2 is not None:
                pil_image2 = ToPILImage()(image2[0].permute(2, 0, 1))
                temp_path2 = Path(folder_paths.temp_directory) / f"temp_image2_{seed}.png"
                pil_image2.save(temp_path2)
                temp_paths.append(temp_path2)

            try:
                # 处理输入图像
                images = []
                if image1 is not None:
                    images.append(pil_image1)
                if image2 is not None:
                    images.append(pil_image2)

                # 准备模型输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    }
                ]

                if images:
                    for img in images:
                        messages[0]["content"].append({"type": "image"})

                # 生成prompt
                prompt = ModelManager._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # 处理输入
                inputs = ModelManager._processor(
                    text=prompt,
                    images=images if images else None,
                    return_tensors="pt",
                ).to(ModelManager._model.device)

                # 生成回答 - 优化推理配置
                start_time = time.time()
                
                # 根据性能模式智能调整推理参数
                if performance_mode == "speed":
                    # 速度优先模式：减少推理步骤，优化性能
                    if temperature == 0:
                        # 温度为0时使用贪婪解码，最快
                        generation_config = {
                            "max_new_tokens": min(max_new_tokens, 1024),  # 限制生成长度
                            "temperature": 0.0,
                            "do_sample": False,
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                    else:
                        # 温度大于0时使用轻量级采样
                        generation_config = {
                            "max_new_tokens": min(max_new_tokens, 1024),  # 限制生成长度
                            "temperature": temperature,
                            "do_sample": True,
                            "top_p": 0.8,  # 降低top_p以减少计算
                            "top_k": 30,   # 降低top_k以减少计算
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                elif performance_mode == "quality":
                    # 质量优先模式：最大化推理质量
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "do_sample": True,
                        "top_p": 0.95,  # 提高top_p以获得更多样化的输出
                        "top_k": 100,   # 增加top_k以获得更多候选
                        "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                    }
                else:  # balanced模式
                    # 平衡模式：根据温度智能选择解码策略
                    if temperature == 0:
                        # 温度为0时使用贪婪解码，更快且确定性
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.0,
                            "do_sample": False,
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                    else:
                        # 温度大于0时使用采样解码
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "do_sample": True,
                            "top_p": 0.9,  # 添加核采样，提高质量
                            "top_k": 50,   # 限制候选词数量
                            "pad_token_id": ModelManager._processor.tokenizer.eos_token_id,
                            "eos_token_id": ModelManager._processor.tokenizer.eos_token_id,
                        }
                
                output_ids = ModelManager._model.generate(
                    **inputs,
                    **generation_config
                )
                inference_time = time.time() - start_time
                print(f"[{self.__class__.__name__}] 推理完成，耗时: {inference_time:.2f}秒")

                # 处理输出
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                result = ModelManager._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            finally:
                # 清理临时文件
                for path in temp_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

            # 将结果存入缓存
            self.update_cache(cache_key, (result, text), use_cache)
            
            print(f"[Qwen3_VQA_Quick] 推理完成")
            return (result, text)
            
        finally:
            # 减少模型引用计数
            ModelManager.release_model(self.session_id)

NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,
    "Qwen3_VQA_Quick": Qwen3_VQA_Quick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",
    "Qwen3_VQA_Quick": "Qwen3 VQA Quick",
}

# 节点加载成功日志
print("[ComfyUI_Qwen3-VL-Instruct] 所有节点加载成功")
