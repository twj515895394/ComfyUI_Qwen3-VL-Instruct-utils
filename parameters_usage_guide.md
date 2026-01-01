# max_batch_size 和 attention 实际使用指南

## 🎯 场景化配置推荐

### 场景1：快速测试和调试
```json
{
  "max_batch_size": 1,
  "attention": "flash_attention_2",
  "performance_mode": "speed"
}
```
- **适用**：单图像分析，快速原型测试
- **优势**：最快速度，最少内存

### 场景2：日常图像问答
```json
{
  "max_batch_size": 1,
  "attention": "flash_attention_2", 
  "performance_mode": "balanced"
}
```
- **适用**：常规图像分析任务
- **优势**：平衡速度和质量

### 场景3：多图像对比分析
```json
{
  "max_batch_size": 2,
  "attention": "flash_attention_2",
  "performance_mode": "quality"
}
```
- **适用**：前后对比、图像差异分析
- **优势**：处理两张图像，获得最佳质量

### 场景4：硬件资源受限
```json
{
  "max_batch_size": 1,
  "attention": "eager",
  "performance_mode": "speed"
}
```
- **适用**：老显卡、显存不足
- **优势**：最大兼容性

## 💡 实际使用示例

### Qwen3_VQA_Quick 节点 - 双图像对比
```
输入：
├── image1: 产品A照片
├── image2: 产品B照片  
├── prompt_template: 产品对比模板
└── max_batch_size: 2

输出：
└── A和B的详细对比分析报告
```

### Qwen3_VQA 节点 - 单图像分析
```
输入：
├── image: 产品照片
├── text: "分析这个产品的特点"
└── max_batch_size: 1

输出：
└── 产品的详细特征分析
```

## ⚠️ 注意事项和限制

### max_batch_size 注意事项
1. **内存需求**：batch_size翻倍，显存需求约翻倍
2. **处理时间**：多图像处理时间会相应增加
3. **质量影响**：batch_size过大可能影响单张图像的分析深度

### attention 注意事项  
1. **兼容性**：flash_attention_2需要CUDA 11.6+和现代显卡
2. **降级方案**：如果flash_attention_2不可用，会自动降级
3. **性能差异**：不同机制的速度差异可达20-30%

## 🔧 故障排除

### 问题1：显存不足
**解决方案：**
```
1. 降低 max_batch_size 到 1
2. 将 attention 改为 "eager" 
3. 使用 performance_mode: "speed"
```

### 问题2：推理速度慢
**解决方案：**
```
1. 确保 attention 设为 "flash_attention_2"
2. 使用 performance_mode: "speed"
3. 降低 max_new_tokens 参数
```

### 问题3：兼容性问题
**解决方案：**
```
1. 将 attention 改为 "eager"
2. 检查CUDA版本是否支持flash_attention_2
3. 考虑升级显卡驱动
```