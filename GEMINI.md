# ComfyUI Qwen3-VL Instruct Utils

## 项目概览

本项目为 **ComfyUI** 提供了一套自定义节点，用于集成 **Qwen3-VL** (及 Qwen2-VL) 视觉语言模型。它允许用户直接在 ComfyUI 工作流中执行多模态任务，例如视觉问答 (VQA)、图像描述和视频分析。

本项目支持两种主要运行模式：
1.  **本地推理 (Local Inference):** 使用 `transformers` 和 `qwen-vl-utils` 在本地运行 Qwen 模型。
2.  **API 推理 (API Inference):** 使用阿里云 DashScope API (兼容 OpenAI SDK) 访问 Qwen-VL 模型 (如 `qwen-vl-plus`, `qwen-vl-max`)，无需本地 GPU 负载。

## 架构

本项目遵循标准的 ComfyUI 自定义节点结构：

*   **入口点:** `__init__.py` 通过 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 注册节点。
*   **核心逻辑:**
    *   **本地节点 (`nodes.py`):** 处理模型加载、缓存 (通过 `ModelManager`) 和推理。使用 `transformers` 加载模型，使用 `qwen_vl_utils` 处理视觉信息。
    *   **API 节点 (`qwen3_vl_api_nodes.py`):** 处理与阿里云 API 的通信。使用 `openai` Python SDK。从本地文件读取 API Key。
*   **工具:**
    *   `util_nodes.py`: 提供用于加载图像和视频的辅助节点 ("Advanced" loaders)。
    *   `text_read_nodes.py` / `text_save_nodes.py`: 处理文本和 Excel 文件读写的工具。
    *   `path_nodes.py`: 处理文件路径的工具。

## 关键文件

*   **`nodes.py`**: 包含本地推理节点的实现：
    *   `Qwen3_VQA`: 标准 VQA 节点。
    *   `Qwen3_VQA_Quick`: 优化/简化版的 VQA 节点。
    *   `ModelManager`: 负责管理模型生命周期、引用计数和缓存的类，以优化 VRAM 使用。
*   **`qwen3_vl_api_nodes.py`**: 包含 API 版本节点的实现：
    *   `Qwen3_VQA_API`: VQA 节点的 API 版本。
    *   `Qwen3_VQA_Quick_API`: Quick VQA 节点的 API 版本。
    *   **配置:** 需要在同一目录下存在 `ali_api_key.txt` 文件。
*   **`util_nodes.py`**: 自定义的图像和视频加载器 (`ImageLoader`, `VideoLoader`)，可能提供比标准 ComfyUI 加载器更多的功能。
*   **`requirements.txt`**: Python 依赖列表 (例如 `torch`, `transformers`, `qwen-vl-utils`, `openai`, `av`)。
*   **`API版节点设计文档.md`**: 详细说明 API 节点设计和使用的文档。

## 安装与配置

1.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    *注意: 请确保您在 ComfyUI 的 Python 环境中执行此操作。在本项目中，建议使用的 Python 执行文件路径为：`X:\ComfyUI-aki-v2\python\python.exe`。*

2.  **API 配置 (可选):**
    如果使用 API 节点，请在当前自定义节点根目录下创建一个名为 `ali_api_key.txt` 的文件，并将您的阿里云 API Key 粘贴进去。

## 使用指南

### 本地推理
*   **模型下载:** 如果模型不存在，通常会在首次使用时自动下载到 `ComfyUI/models/prompt_generator/` (或标准的 HuggingFace 缓存目录)。
*   **内存管理:** `nodes.py` 中的 `ModelManager` 处理加载/卸载。节点中的 `keep_model_loaded` 参数控制生成后模型是否保留在显存中。

### API 推理
*   **节点:** 使用 `Qwen3 VQA API` 或 `Qwen3 VQA Quick API`。
*   **成本:** 会在阿里云平台上产生费用。
*   **兼容性:** 设计为与本地节点的输入/输出匹配，以便在工作流中轻松替换。

## 开发规范

*   **节点映射:** 所有新节点必须在 `__init__.py` 中注册。
*   **类型提示:** ComfyUI 的 `INPUT_TYPES` 和 `RETURN_TYPES` 定义非常严格，需仔细核对。
*   **错误处理:** API 节点应优雅地处理网络错误和缺失 API Key 的情况。
*   **风格:** 遵循标准的 Python PEP 8 规范。