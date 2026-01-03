# Qwen3-VL 批量反推节点设计与测试方案

## 1. 需求背景
用户需要对大量图片进行视觉反推（Captioning）。目前使用 ComfyUI 的 List 循环调用单图节点，效率低下（单图耗时 20-30s）。
目标是开发一个 `Qwen3_VQA_Batch` 节点，接收图片列表，一次性输出对应的提示词列表，显著提升吞吐量（Throughput）。

## 2. 技术方案分析

### 方案 A：True Batching (并行推理) - **首选**
利用 `transformers` 的 batch generation 能力。
*   **原理**: 构造 `[Batch_Size, Sequence_Length]` 的输入张量。
*   **难点**: Qwen-VL 使用动态分辨率（Naive Dynamic Resolution），每张图片的 Vision Token 数量可能不同。这意味着需要大量的 Padding (Left Padding for generation)，这会浪费计算资源，甚至在某些实现下导致 Attention Mask 错误。
*   **对策**: 
    1.  强制将 Batch 内的图片 Resize 到相同的分辨率（或相近的 Patch 数），减少 Padding 浪费。
    2.  设置 `max_batch_size` 限制，防止显存爆炸。

### 方案 B：In-Node Loop (节点内循环) - **备选**
在 Python 节点内部对图片列表进行 `for` 循环推理。
*   **原理**: 串行处理，但省去了 ComfyUI 的调度开销和模型重复 Check 开销。
*   **优势**: 实现简单，显存占用稳定（一次只处理一张）。
*   **劣势**: 无法利用 GPU 并行能力，加速上限低。

**结论**: 我们将优先尝试 **方案 A (True Batching)**。如果遇到难以逾越的技术障碍（如 Qwen2VL 库对 Batch 支持的 Bug），则回退到 **方案 B** 并通过多线程数据预处理优化。

## 3. 节点设计: `Qwen3_VQA_Batch`

### 3.1 输入参数 (INPUT_TYPES)
*   **required**:
    *   `images`: (`IMAGE`,) - ComfyUI 的图像批次张量 `[B, H, W, C]`。
    *   `prompt_template`: (File) - 提示词模板。
    *   `model`, `quantization`, `keep_model_loaded`... (同上)
    *   `batch_size`: (`INT`, default=4) - 内部推理的实际 Batch Size。如果输入图片数 > batch_size，则分批执行。
*   **optional**:
    *   `user_prompt`: (`STRING`) - 辅助提示词。

### 3.2 输出 (RETURN_TYPES)
*   `captions`: (`LIST[STRING]`) - 对应每张图片的文本列表。
    *   *注意*: ComfyUI 默认不支持直接返回 List[String] 给某些节点，可能需要配合 `Batch List` 相关的辅助节点使用，或者将其拼接成一个大字符串返回。为了通用性，我们返回 List，但 ComfyUI 前端可能会显示为 Batch 形式。

## 4. 性能测试方案

### 4.1 测试环境
*   Python: `X:\ComfyUI-aki-v2\python\python.exe`
*   GPU: 用户当前环境
*   测试图片: `test_pic/` 目录下复制多张图片 (准备 4-8 张)。

### 4.2 测试脚本 (`run_batch_benchmark.py`)
脚本将对比两种模式的耗时：
1.  **Baseline (Loop Mode)**: 模拟 ComfyUI 外部循环，连续调用 `Qwen3_VQA_Quick` 4次。
2.  **Batch Mode**: 调用 `Qwen3_VQA_Batch` 一次，输入 4 张图片。

### 4.3 评估指标
*   **Total Latency**: 总耗时。
*   **Avg Latency per Image**: 单图平均耗时。
*   **Peak VRAM**: 显存峰值（如果在 CLI 难以测量，则通过观察任务管理器估算，或只关注是否 OOM）。

## 5. 开发计划
1.  **准备数据**: 在 `test_pic` 中复制图片，构造 `test_1.jpg`, `test_2.jpg`...
2.  **实现节点**: 修改 `nodes.py`，增加 `Qwen3_VQA_Batch` 类。核心逻辑是构造 Batch Inputs。
3.  **编写测试**: 实现对比脚本。
4.  **调优**: 根据测试结果调整 `padding_side` (通常 LLM generate 需要 left padding) 和 `batch_size` 默认值。
