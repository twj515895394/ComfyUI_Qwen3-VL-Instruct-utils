import os
import torch
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


class Qwen3_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.current_model_id = None  # Track the current model id
        self.current_quantization = None  # Track the current quantization
        self.cache = {}  # 用于存储输入参数和模型输出的缓存
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
                ),
                "use_cache": ("BOOLEAN", {"default": True}),  # 缓存开关
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
            attention="eager",
            use_cache=True,
    ):
        # 创建缓存键
        cache_key = self._create_cache_key(
            text=text,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            seed=seed,
            quantization=quantization,
            source_path=source_path,
            image=image,
            attention=attention
        )
        
        # 检查是否可以从缓存获取结果
        if use_cache:
            print(f"[Qwen3_VQA] 缓存开关已开启，检查是否存在缓存结果")
            if cache_key in self.cache:
                print(f"[Qwen3_VQA] 缓存命中！直接返回缓存结果")
                return (self.cache[cache_key],)
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

        # If model_id or quantization changed, reload processor and model
        if (
                self.current_model_id != model_id
                or self.current_quantization != quantization
                or self.processor is None
                or self.model is None
        ):
            self.current_model_id = model_id
            self.current_quantization = quantization
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )
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

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

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
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            # Inference: Generation of the output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.processor  # release processor memory
                del self.model  # release model memory
                self.processor = None  # set processor to None
                self.model = None  # set model to None
                self.current_model_id = None
                self.current_quantization = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release GPU memory
                    torch.cuda.ipc_collect()

            # 将结果存入缓存
            if use_cache:
                self.cache[cache_key] = result
                print(f"[Qwen3_VQA] 结果已存入缓存，当前缓存大小: {len(self.cache)}")
            
            print(f"[Qwen3_VQA] 推理完成")
            return (result,)
    
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


class Qwen3_VQA_Quick:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.current_model_id = None  # Track the current model id
        self.current_quantization = None  # Track the current quantization
        self.cache = {}  # 用于存储输入参数和模型输出的缓存
        # 提示词模板文件夹路径
        self.prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        # 确保提示词文件夹存在
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir)
        print("[Qwen3_VQA_Quick] 节点初始化完成，缓存系统已设置")

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
                ),
                "use_cache": ("BOOLEAN", {"default": True}),  # 缓存开关
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
            attention="eager",
            use_cache=True,
    ):
        # 创建缓存键
        cache_key = self._create_cache_key(
            prompt_template=prompt_template,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            seed=seed,
            quantization=quantization,
            source_path=source_path,
            image1=image1,
            image2=image2,
            attention=attention
        )
        
        # 检查是否可以从缓存获取结果
        if use_cache:
            print(f"[Qwen3_VQA_Quick] 缓存开关已开启，检查是否存在缓存结果")
            if cache_key in self.cache:
                print(f"[Qwen3_VQA_Quick] 缓存命中！直接返回缓存结果")
                result, text = self.cache[cache_key]
                print(f"[Qwen3_VQA_Quick] 推理完成")
                return (result, text)
        
        # 读取提示词模板内容
        template_text = self.read_prompt_template(prompt_template)
        # 拼接用户输入的辅助提示词
        text = template_text
        if user_prompt.strip():
            text += f"\n\nUser prompt:{user_prompt}"
        
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

        # If model_id or quantization changed, reload processor and model
        if (
                self.current_model_id != model_id
                or self.current_quantization != quantization
                or self.processor is None
                or self.model is None
        ):
            self.current_model_id = model_id
            self.current_quantization = quantization
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )
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

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

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
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 处理输入
            inputs = self.processor(
                text=prompt,
                images=images if images else None,
                return_tensors="pt",
            ).to(self.model.device)

            # 生成回答
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

            # 处理输出
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        finally:
            # 清理临时文件
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

        # 不保留模型时清理
        if not keep_model_loaded:
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        # 将结果存入缓存
        if use_cache:
            self.cache[cache_key] = (result, text)
            print(f"[Qwen3_VQA_Quick] 结果已存入缓存，当前缓存大小: {len(self.cache)}")
        
        print(f"[Qwen3_VQA_Quick] 推理完成")
        return (result, text)

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
