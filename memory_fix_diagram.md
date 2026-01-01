# Qwen3_VQA_Quick节点内存溢出修复方案

## 问题分析

### 原始设计问题
```mermaid
graph TD
    A[用户循环调用Qwen3_VQA_Quick节点] --> B[每个节点实例检查keep_model_loaded]
    B --> C[keep_model_loaded=True]
    C --> D[所有实例共享类变量<br/>self.model和self.processor]
    D --> E[最后一个调用的实例释放内存]
    E --> F[早期实例仍在使用已释放的模型]
    F --> G[内存访问错误和OOM]
```

### 根本原因
1. **类级别共享与实例级别控制的冲突**
2. **循环调用时内存释放影响其他节点** 
3. **内存碎片化**

## 修复方案

### 新的设计架构
```mermaid
classDiagram
    class ModelManager {
        -_model: Qwen3VLForConditionalGeneration
        -_processor: AutoProcessor
        -_current_model_id: str
        -_current_quantization: str
        -_reference_count: int
        +acquire_model(session_id)
        +release_model(session_id)
        +get_model_info()
        +_cleanup_if_needed()
    }
    
    class Qwen3_Base {
        -session_id: str
        -_cache: dict
        +generate_cache_key_without_random_seed()
        +check_cache()
        +update_cache()
    }
    
    class Qwen3_VQA_Quick {
        +inference()
    }
    
    Qwen3_Base --> ModelManager : uses
    Qwen3_VQA_Quick --> Qwen3_Base : inherits
```

### 修复后的工作流程
```mermaid
sequenceDiagram
    participant U as 用户
    participant N1 as 节点实例1
    participant N2 as 节点实例2
    participant M as ModelManager
    
    U->>N1: 第一次调用
    N1->>M: acquire_model()
    M->>M: 增加引用计数=1
    M->>M: 加载模型和处理器
    N1->>M: 获取模型信息
    N1->>M: 执行推理
    N1->>M: release_model()
    M->>M: 减少引用计数=0
    M->>M: 释放模型内存
    
    U->>N2: 第二次调用
    N2->>M: acquire_model()
    M->>M: 增加引用计数=1
    M->>M: 重新加载模型
    N2->>M: 获取模型信息
    N2->>M: 执行推理
    N2->>M: release_model()
    M->>M: 减少引用计数=0
    M->>M: 释放模型内存
```

## 关键改进点

### 1. 引用计数机制
```mermaid
graph LR
    A[acquire_model] --> B[reference_count += 1]
    B --> C{reference_count > 0?}
    C -->|是| D[保持模型加载]
    C -->|否| E[释放模型内存]
    
    F[release_model] --> G[reference_count -= 1]
    G --> H{reference_count == 0?}
    H -->|是| E
    H -->|否| I[等待其他实例释放]
```

### 2. 会话隔离
- 每个节点实例有唯一的session_id
- 避免实例间的相互影响
- 支持并发调用

### 3. 内存管理优化
```mermaid
graph TD
    A[推理开始] --> B[acquire_model]
    B --> C[检查模型状态]
    C --> D{需要重新加载?}
    D -->|是| E[清理旧模型]
    D -->|否| F[复用现有模型]
    E --> G[加载新模型]
    F --> H[执行推理]
    G --> H
    H --> I[释放引用计数]
    I --> J[推理结束]
```

## 预期效果

### 性能提升
1. **消除OOM**：引用计数确保模型只在真正无人使用时才释放
2. **减少重复加载**：相同配置的请求可以复用已加载的模型
3. **提高并发性能**：支持多个节点实例同时工作

### 内存使用优化
- **智能缓存**：只有在引用计数为0时才清理内存
- **按需加载**：每个请求都会正确管理模型生命周期
- **避免内存泄漏**：确保所有资源都能正确释放

## 测试建议

### 循环调用测试
```python
# 测试场景：连续调用10次Qwen3_VQA_Quick节点
for i in range(10):
    result = qwen_node.inference(
        prompt_template="test.txt",
        model="Huihui-Qwen3-VL-8B-Instruct-abliterated",
        keep_model_loaded=True,
        # ... 其他参数
    )
    print(f"第{i+1}次调用完成")
```

### 预期结果
- 内存使用稳定，不会持续增长
- 只有第一次和最后一次调用会重新加载模型
- 中间的调用可以复用已加载的模型