import os
import torch
import time
import folder_paths
import random
import gc
import io
import hashlib
import json
import base64
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import comfy.model_management
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 1. 辅助函数与常量
# ==============================================================================

def validate_seed(seed):
    """验证和修复seed种子值"""
    MAX_SEED = 2**31 - 1
    is_random = False
    if seed == -1 or seed > MAX_SEED:
        seed = random.randint(0, MAX_SEED)
        is_random = True
    elif seed < 0:
        seed = 0
    return seed, is_random

def get_model_path(model_name):
    """获取模型路径，优先查找 prompt_generator 目录"""
    # 优先查找 prompt_generator 目录
    model_dir = os.path.join(folder_paths.models_dir, "prompt_generator")
    
    # 特殊处理 Huihui 模型 - 兼容原始路径
    if model_name == "Huihui-Qwen3-VL-8B-Instruct-abliterated":
        huihui_path = f"Y:\\llama-models\\Qwen3-vl-nsfw\\prompt_generator\\Huihui-Qwen3-VL-8B-Instruct-abliterated"
        if os.path.exists(huihui_path):
            return huihui_path, os.path.dirname(huihui_path)
        target_path = os.path.join(model_dir, model_name)
    else:
        # 处理 qwen/Qwen3... 格式
        target_path = os.path.join(model_dir, model_name)
    
    # 如果路径不存在，尝试从 HuggingFace 格式推断
    if not os.path.exists(target_path):
        # 兼容性处理：如果模型名包含 repo 前缀，取 basename
        model_base = os.path.basename(model_name)
        alt_path = os.path.join(model_dir, model_base)
        if os.path.exists(alt_path):
            return alt_path, model_dir
        
    return target_path, model_dir

# ==============================================================================
# 2. 策略模式: Attention 机制管理
# ==============================================================================

class AttentionStrategy:
    """管理不同的 Attention 实现策略"""
    
    @staticmethod
    def apply_patch(strategy_name):
        """应用 Attention Patch (主要是 SageAttention)"""
        if strategy_name == "sage_attention_2":
            try:
                import sageattention
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, apply_multimodal_rotary_pos_emb
                
                print("[AttentionStrategy] 正在应用 SageAttention2 补丁...")
                
                def sage_attention_forward(self, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, 
                                        past_key_values=None, output_attentions: bool = False, use_cache: bool = False, 
                                        cache_position=None, position_embeddings=None, **kwargs):
                    bsz, q_len, _ = hidden_states.size()
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)
                    
                    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                    
                    cos, sin = position_embeddings
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
                    )
                    
                    if past_key_values is not None:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                    # NHD Layout for SageAttention
                    query_states = query_states.transpose(1, 2)
                    key_states = key_states.transpose(1, 2)
                    value_states = value_states.transpose(1, 2)
                    
                    attn_output = sageattention.sageattn(
                        query_states, key_states, value_states,
                        is_causal=self.is_causal, tensor_layout="NHD"
                    )
                    
                    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
                    attn_output = self.o_proj(attn_output)
                    return attn_output, None

                Qwen2VLAttention.forward = sage_attention_forward
                print("[AttentionStrategy] SageAttention2 补丁应用成功。")
                return "sdpa" # SageAttn works on top of SDPA base in this context usually
            except ImportError:
                print("[AttentionStrategy] 未找到 SageAttention 库。回退到 FlashAttention2。")
                return "flash_attention_2"
            except Exception as e:
                print(f"[AttentionStrategy] SageAttention2 补丁应用失败: {e}。回退到默认设置。")
                return "eager"
        
        return strategy_name

    @staticmethod
    def get_implementation(requested_attention):
        """获取实际可用的 Attention 实现名称"""
        if requested_attention == "flash_attention_2":
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                print("[AttentionStrategy] FlashAttention2 不可用。回退到 sdpa。")
                return "sdpa"
        if requested_attention == "sage_attention_2":
             # 如果 patch 没成功（没 return sdpa），这里也会被调用
             return "sdpa"
        return requested_attention

# ==============================================================================
# 3. 单例模式: Model Manager
# ==============================================================================

class Qwen3ModelManager:
    """单例类：负责模型的加载、缓存与生命周期管理"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Qwen3ModelManager, cls).__new__(cls)
            cls._instance._init_resources()
        return cls._instance

    def _init_resources(self):
        self.model = None
        self.processor = None
        self.current_model_id = None
        self.current_quantization = None
        self.current_attention = None
        self.reference_count = 0
        self.active_sessions = set()
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available() and 
            torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    def acquire(self, session_id):
        self.active_sessions.add(session_id)
        self.reference_count += 1
        # print(f"[ModelManager] Acquire: {self.reference_count} refs")

    def release(self, session_id, keep_loaded=False):
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            self.reference_count = max(0, self.reference_count - 1)
            # print(f"[ModelManager] Release: {self.reference_count} refs")
            
            if self.reference_count == 0 and not keep_loaded:
                self.unload_all()

    def unload_all(self):
        print("[ModelManager] 正在释放模型资源...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.current_model_id = None
        self.current_quantization = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print("[ModelManager] 模型资源已释放。")

    def load_model(self, model_name, quantization, attention_mode, min_pixels, max_pixels):
        """加载或重载模型"""
        # 1. 确定模型ID和路径
        if model_name == "Huihui-Qwen3-VL-8B-Instruct-abliterated":
            repo_id = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        else:
            repo_id = f"qwen/{model_name}"
            
        model_path, model_parent_dir = get_model_path(model_name)
        
        # 自动下载逻辑
        if not os.path.exists(model_path):
            print(f"[ModelManager] 模型未在路径找到: {model_path}。正在下载...")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_path,
                    allow_patterns=["*.json", "*.bin", "*.model", "*.pth", "*.safetensors"],
                )
            except Exception as e:
                print(f"[ModelManager] 下载失败: {e}")
                raise RuntimeError(f"Could not download model {repo_id}")

        # 2. 检查是否需要重新加载
        # 注意：这里我们加入对 min/max pixels 的检查，因为 processor 需要它们
        # 但为了避免因为像素参数微调导致重载模型(非常耗时)，我们只在 Processor 层面处理
        # 真正的重模型加载只看 model_id, quantization, attention
        
        actual_attention = AttentionStrategy.apply_patch(attention_mode)
        actual_attention = AttentionStrategy.get_implementation(actual_attention)

        # 检查是否可以复用模型权重
        if (self.model is not None and 
            self.current_model_id == repo_id and 
            self.current_quantization == quantization and
            self.current_attention == actual_attention):
            print("[ModelManager] 复用已加载的模型。")
            
            # 即使模型复用，Processor 可能需要更新 min/max pixels
            # 这是一个轻量级操作，我们总是重新加载 Processor 以确保参数生效
            # 或者我们可以检查 processor 的配置，但重新加载 Processor 很快 (<1s)
            self.processor = AutoProcessor.from_pretrained(
                model_path, min_pixels=min_pixels, max_pixels=max_pixels
            )
            return

        # 3. 加载新模型
        print(f"[ModelManager] 正在加载模型: {repo_id} (Quant: {quantization}, Attn: {actual_attention})")
        
        # 卸载旧模型
        self.unload_all()

        start_time = time.time()
        
        # 配置 Quantization
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if self.bf16_support else torch.float16)
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            # 加载 Processor
            self.processor = AutoProcessor.from_pretrained(
                model_path, min_pixels=min_pixels, max_pixels=max_pixels
            )

            # 加载 Model
            # 注意: Qwen3-VL 代码库通常兼容 Qwen2VL 类
            # 如果 transformers 版本较新，可以直接用 Qwen2VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=actual_attention,
                quantization_config=bnb_config,
            )
            
            self.current_model_id = repo_id
            self.current_quantization = quantization
            self.current_attention = actual_attention
            
            print(f"[ModelManager] 模型加载完成，耗时: {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"[ModelManager] 加载错误: {e}")
            self.unload_all()
            raise e

# ==============================================================================
# 4. 模板方法模式: Base Node
# ==============================================================================

class Qwen3BaseNode:
    """所有 Qwen3 节点的基类"""
    
    def __init__(self):
        self.session_id = str(id(self))
        self.cache = {}
        self.max_cache_size = 50
        self.manager = Qwen3ModelManager.get_instance()

    def _get_cache_key(self, **kwargs):
        """生成高效的缓存键"""
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if k == "seed" and v == -1: continue # 忽略随机种子占位符
            
            if isinstance(v, torch.Tensor):
                # 优化: 仅计算 shape 和少量数据的 hash，避免全量计算
                meta = f"{v.shape}-{v.device}-{v.dtype}"
                # 采样中心点
                if v.numel() > 0:
                    center_val = v.view(-1)[v.numel() // 2].item()
                    meta += f"-{center_val:.4f}"
                key_parts.append(f"{k}:{meta}")
            else:
                key_parts.append(f"{k}:{str(v)}")
        
        raw_key = "|".join(key_parts)
        return hashlib.md5(raw_key.encode()).hexdigest()

    def _tensor_to_pil(self, image_tensor):
        """将 ComfyUI Tensor (B,H,W,C) 转换为 PIL Image 列表"""
        if image_tensor is None:
            return []
        
        pil_images = []
        # 遍历 Batch
        for i in range(image_tensor.shape[0]):
            img = image_tensor[i] # [H, W, C]
            # 确保是 CPU
            img = img.cpu().numpy()
            # 转换为 PIL (需要先转为 uint8 0-255)
            img = (img * 255).clip(0, 255).astype("uint8")
            pil_images.append(Image.fromarray(img))
        return pil_images

    def _pil_to_base64(self, pil_image):
        """内存中转换 PIL 为 Base64，避免磁盘 IO"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def inference_template(self, 
                          prompt_struct_func,  # 回调：构建 prompt 结构
                          model_name, keep_model_loaded, temperature, max_new_tokens,
                          min_pixels, max_pixels, seed, quantization, attention,
                          performance_mode, **kwargs):
        
        # 1. 缓存检查
        cache_key = self._get_cache_key(
            model=model_name, temp=temperature, tokens=max_new_tokens,
            min_px=min_pixels, max_px=max_pixels, seed=seed, quant=quantization,
            attn=attention, perf=performance_mode, **kwargs
        )
        if cache_key in self.cache:
            print(f"[{self.__class__.__name__}] 缓存命中!")
            return self.cache[cache_key]

        # 2. 资源获取
        self.manager.acquire(self.session_id)
        
        try:
            # 3. 模型加载
            self.manager.load_model(model_name, quantization, attention, min_pixels, max_pixels)
            model = self.manager.model
            processor = self.manager.processor
            
            # 4. 设置随机种子
            actual_seed, _ = validate_seed(seed)
            torch.manual_seed(actual_seed)

            # 5. 处理输入并构建 Prompt
            # kwargs 包含 source_path, images 等
            messages = prompt_struct_func(**kwargs)

            # 6. 预处理 (Use process_vision_info + apply_chat_template)
            # 关键优化：process_vision_info 通常需要 path 或者 base64
            # 我们在 prompt_struct_func 中处理了 Image -> PIL -> Base64/Object 的转换
            
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.manager.device)

            # 7. 生成配置 (Performance Mode)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            
            if performance_mode == "speed":
                # 速度模式：Greedy 或低采样
                if temperature > 0:
                    gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.8, "top_k": 20})
                else:
                    gen_kwargs.update({"do_sample": False})
            elif performance_mode == "quality":
                # 质量模式
                gen_kwargs.update({"do_sample": True, "temperature": max(0.1, temperature), "top_p": 0.95, "top_k": 100})
            else: # balanced
                if temperature == 0:
                    gen_kwargs.update({"do_sample": False})
                else:
                    gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9, "top_k": 50})

            # 8. 推理
            start_t = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)
            
            inference_time = time.time() - start_t
            print(f"[{self.__class__.__name__}] 推理完成，耗时: {inference_time:.2f}s")

            # 9. 解码
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 10. 缓存与返回
            # 对于 Qwen3_VQA_Quick，我们可能需要返回 (output, prompt_text)
            # 这里我们让调用者处理返回值格式，template 返回纯文本
            
            self.cache[cache_key] = output_text
            if len(self.cache) > self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            
            return output_text, text_prompt

        finally:
            self.manager.release(self.session_id, keep_model_loaded)


# ==============================================================================
# 5. 具体节点实现
# ==============================================================================

class Qwen3_VQA(Qwen3BaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "输入给模型的文本提示词或问题。"}),
                "model": ([
                    "Huihui-Qwen3-VL-8B-Instruct-abliterated",
                    "Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct",
                    "Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking",
                    "Qwen3-VL-4B-Instruct-FP8", "Qwen3-VL-8B-Instruct-FP8"
                ], {"default": "Huihui-Qwen3-VL-8B-Instruct-abliterated", "tooltip": "选择要使用的 Qwen3-VL 模型版本。"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none", "tooltip": "选择模型量化方式。4bit/8bit 可以显著降低显存占用，但可能略微降低质量。"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "如果开启，模型在推理完成后将保留在显存中，下次运行将非常快。"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1, "tooltip": "控制生成文本的随机性。0 为确定性输出，值越高结果越随机。"}),
                "max_new_tokens": ("INT", {"default": 768, "min": 128, "max": 32000, "tooltip": "限制模型输出的最大文本长度（Token 数量）。值越大生成越慢。"}), 
                "min_pixels": ("INT", {"default": 256 * 28 * 28, "min": 64*28*28, "tooltip": "处理图片时的最小总像素数（单位：像素点）。"}),
                "max_pixels": ("INT", {"default": 768 * 28 * 28, "min": 256*28*28, "tooltip": "处理图片时的最大总像素数。超过此值的图片会被自动缩小，这直接影响推理速度和视觉细节。"}), 
                "seed": ("INT", {"default": -1, "tooltip": "随机种子。-1 表示每次随机生成。"}),
                "attention": (["sage_attention_2", "flash_attention_2", "sdpa", "eager"], {"default": "sage_attention_2", "tooltip": "选择注意力机制实现。sage_attention_2 通常最快，sdpa 为系统默认优化。"}),
                "performance_mode": (["balanced", "speed", "quality"], {"default": "balanced", "tooltip": "性能模式预设。speed 优先速度，quality 优先质量。"}),
            },
            "optional": {
                "source_path": ("PATH",), 
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def run_inference(self, text, model, keep_model_loaded, temperature, max_new_tokens,
                     min_pixels, max_pixels, seed, quantization, source_path=None, image=None,
                     attention="sage_attention_2", performance_mode="balanced", **kwargs):
        
        def build_prompt(source_path=None, image=None, **kwargs):
            content = []
            
            # 处理图像
            if image is not None:
                pil_images = self._tensor_to_pil(image)
                for img in pil_images:
                    content.append({"type": "image", "image": img})
            
            # 处理 Path (假设是视频或图片路径列表)
            if source_path:
                # source_path 可能是 list 或 str，这里假设上游传入的是兼容 process_vision_info 的格式
                if isinstance(source_path, list):
                    content.extend(source_path)
                else:
                    # 简单字符串路径
                    # TODO: 检测是视频还是图片
                    content.append({"type": "image", "image": source_path})

            content.append({"type": "text", "text": text})
            
            return [{"role": "user", "content": content}]

        result_text, _ = self.inference_template(
            build_prompt, model, keep_model_loaded, temperature, max_new_tokens,
            min_pixels, max_pixels, seed, quantization, attention, performance_mode,
            source_path=source_path, image=image
        )
        return (result_text,)

class Qwen3_VQA_Quick(Qwen3BaseNode):
    def __init__(self):
        super().__init__()
        self.prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

    @classmethod
    def INPUT_TYPES(s):
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        prompt_files = []
        if os.path.exists(prompts_dir):
            prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith((".txt", ".md", ".json"))]
        
        return {
            "required": {
                "prompt_template": (sorted(prompt_files), {"default": prompt_files[0] if prompt_files else "", "tooltip": "选择提示词模板文件。"}),
                "model": ([
                    "Huihui-Qwen3-VL-8B-Instruct-abliterated",
                    "Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct",
                    "Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking",
                    "Qwen3-VL-4B-Instruct-FP8", "Qwen3-VL-8B-Instruct-FP8"
                ], {"default": "Huihui-Qwen3-VL-8B-Instruct-abliterated", "tooltip": "选择要使用的 Qwen3-VL 模型版本。"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入的辅助提示词，会拼接在模板内容之后。"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none", "tooltip": "选择模型量化方式。4bit/8bit 可以显著降低显存占用。"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "如果开启，模型在推理完成后将保留在显存中。"}),
                "temperature": ("FLOAT", {"default": 0.7, "tooltip": "控制生成文本的随机性。"}),
                "max_new_tokens": ("INT", {"default": 768, "tooltip": "限制模型输出的最大文本长度。"}),
                "min_pixels": ("INT", {"default": 256 * 28 * 28, "tooltip": "处理图片时的最小总像素数。"}),
                "max_pixels": ("INT", {"default": 768 * 28 * 28, "tooltip": "处理图片时的最大总像素数。超过此值的图片会被自动缩小，显著影响速度。"}), # Optimized default
                "seed": ("INT", {"default": -1, "tooltip": "随机种子。-1 表示每次随机生成。"}),
                "attention": (["sage_attention_2", "flash_attention_2", "sdpa", "eager"], {"default": "sage_attention_2", "tooltip": "选择注意力机制实现。"}),
                "performance_mode": (["balanced", "speed", "quality"], {"default": "balanced", "tooltip": "性能模式预设。"}),
            },
            "optional": {
                "source_path": ("PATH",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "prompt_text")
    FUNCTION = "run_inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def run_inference(self, prompt_template, model, keep_model_loaded, temperature, max_new_tokens,
                     min_pixels, max_pixels, seed, quantization, user_prompt="", 
                     source_path=None, image1=None, image2=None,
                     attention="sage_attention_2", performance_mode="balanced", **kwargs):

        # 读取模板
        template_content = ""
        try:
            with open(os.path.join(self.prompts_dir, prompt_template), 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception:
            template_content = ""

        def build_prompt(prompt_template_text=None, user_prompt_text=None, source_path=None, image1=None, image2=None, **kwargs):
            # 拼接文本
            final_text = prompt_template_text
            if user_prompt_text and user_prompt_text.strip():
                final_text += f"\n\nUser prompt:{user_prompt_text}"
            
            content = []
            
            # 处理图片 (内存中转 PIL)
            images = []
            if image1 is not None: images.extend(self._tensor_to_pil(image1))
            if image2 is not None: images.extend(self._tensor_to_pil(image2))
            
            for img in images:
                content.append({"type": "image", "image": img})

            if source_path:
                 if isinstance(source_path, list):
                    content.extend(source_path)
                 else:
                    content.append({"type": "image", "image": source_path})

            content.append({"type": "text", "text": final_text})
            
            return [{"role": "user", "content": content}]

        result_text, full_prompt_text = self.inference_template(
            build_prompt, model, keep_model_loaded, temperature, max_new_tokens,
            min_pixels, max_pixels, seed, quantization, attention, performance_mode,
            prompt_template_text=template_content, user_prompt_text=user_prompt,
            source_path=source_path, image1=image1, image2=image2
        )
        
        return (result_text, full_prompt_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,
    "Qwen3_VQA_Quick": Qwen3_VQA_Quick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",
    "Qwen3_VQA_Quick": "Qwen3 VQA Quick",
}