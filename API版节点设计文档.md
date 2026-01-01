# Qwen3-VL API版节点设计文档

## 1. 设计目标
创建API版的Qwen3_VQA和Qwen3_VQA_Quick节点，调用阿里云qwen3-vl-plus大模型，功能与本地版保持一致，支持纯文本和视觉理解。

## 2. API调用方式

### 2.1 API Key管理
API Key将保存在`ali_api_key.txt`文件中，节点将自动读取该文件获取API Key。

### 2.2 调用方式
阿里云qwen3-vl-plus API兼容OpenAI API格式，因此可以使用OpenAI SDK进行调用。

### 2.3 模型信息
- 模型名称：qwen-vl-plus
- 支持模态：文本、图像
- 最大上下文长度：根据阿里云文档确定

## 3. 输入参数格式

### 3.1 公共参数
所有API版节点将支持与本地版相同的公共参数：
- prompt：用户输入的文本提示
- temperature：生成温度（0-1）
- max_new_tokens：最大生成token数
- top_p：核采样参数
- frequency_penalty：频率惩罚
- presence_penalty：存在惩罚
- seed：随机种子

### 3.2 节点特殊输入格式
两个API版节点将根据各自的功能特点，使用不同的输入格式处理：

#### 3.2.1 Qwen3_VQA_API节点输入
该节点用于处理普通的多模态输入，支持文本和图像组合：

**单图像输入**
```json
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片里有什么？"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64_encoded_image"}}
        ]
    }
]
```

**多图像输入**
```json
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "比较这两张图片"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64_encoded_image1"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64_encoded_image2"}}
        ]
    }
]
```

**纯文本输入**（无图像时省略image_url）
```json
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请解释一下量子力学"}
        ]
    }
]
```

#### 3.2.2 Qwen3_VQA_Quick_API节点输入
该节点用于处理模板化输入，要求prompt参数为JSON格式字符串，包含两个字段：
- `system_prompt`：模板的提示词内容
- `user_prompt`：用户输入信息（可选，可为空字符串）

**模板化输入示例**
```json
// prompt参数内容（JSON字符串）
{
    "system_prompt": "请根据用户输入的内容，以清晰的结构回答问题",
    "user_prompt": "这张图片里有什么？"
}
```

**API请求组装结果**
```json
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "{
    \"system_prompt\": \"请根据用户输入的内容，以清晰的结构回答问题\",
    \"user_prompt\": \"这张图片里有什么？\"
}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64_encoded_image"}}
        ]
    }
]
```

### 3.3 图像参数说明
- `image_url`为可选参数，无图像输入时可省略
- 支持单张或多张图像输入（具体数量限制以阿里云API文档为准）
- 图像需转换为base64编码格式

## 4. 输出参数格式
返回的响应格式与OpenAI API一致，主要包含以下字段：

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "qwen-vl-plus",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "这张图片里有一只猫"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
    }
}
```

## 5. 实现思路

### 5.1 节点结构设计
- 创建两个新节点：Qwen3_VQA_API和Qwen3_VQA_Quick_API
- 继承本地版节点的核心功能
- 修改模型加载和推理部分为API调用

### 5.2 API调用流程
1. 从`ali_api_key.txt`读取API Key
2. 构建符合OpenAI API格式的请求参数
3. 将图像转换为base64编码
4. 发送API请求到阿里云服务器
5. 解析API响应并返回结果

### 5.3 与本地版的兼容性
- 保持相同的输入输出接口
- 支持相同的参数配置
- 保持相同的缓存机制（可选）

## 6. 代码结构

```python
# qwen3_vl_api_nodes.py

import os
import base64
import requests
from PIL import Image
import io

# 从文件中读取API Key
def get_api_key():
    with open('ali_api_key.txt', 'r') as f:
        return f.read().strip()

# 图像编码为base64
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class Qwen3_VQA_API:
    """API版Qwen3_VQA节点"""
    @classmethod
    def INPUT_TYPES(s):
        # 保持与本地版相同的输入类型定义
        pass
    
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"
    
    def inference(self, **kwargs):
        api_key = get_api_key()
        # 构建API请求
        # 发送请求并获取响应
        # 解析响应并返回结果
        pass

class Qwen3_VQA_Quick_API:
    """API版Qwen3_VQA_Quick节点"""
    @classmethod
    def INPUT_TYPES(s):
        # 保持与本地版相同的输入类型定义
        pass
    
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"
    
    def inference(self, **kwargs):
        api_key = get_api_key()
        # 构建API请求
        # 发送请求并获取响应
        # 解析响应并返回结果
        pass
```

## 7. 注意事项

1. API Key的安全性：确保`ali_api_key.txt`文件权限设置正确，仅允许节点读取
2. 图像大小限制：阿里云API可能对图像大小有限制，需在代码中添加检查
3. 网络连接：API调用需要稳定的网络连接
4. 错误处理：需添加适当的错误处理机制，处理API调用失败的情况
5. 费用控制：需根据阿里云API定价控制调用次数和输入输出token数

## 8. 下一步计划

1. 实现API版节点代码
2. 测试API调用功能
3. 确保与本地版节点的兼容性
4. 文档更新和完善