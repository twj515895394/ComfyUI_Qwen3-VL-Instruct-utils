# Attention 机制详解

## 三种注意力机制对比

### 1. eager（基础注意力）
- **速度**：中等
- **内存**：较少
- **兼容性**：最好
- **适用场景**：老设备、显存不足
- **特点**：标准实现，稳定可靠

### 2. sdpa（Scaled Dot-Product Attention）
- **速度**：较快
- **内存**：中等
- **兼容性**：较好
- **适用场景**：大多数设备
- **特点**：优化后的注意力计算

### 3. flash_attention_2（推荐）
- **速度**：最快
- **内存**：最少
- **兼容性**：较好
- **适用场景**：现代GPU设备
- **特点**：显存友好的注意力计算

## 使用建议

### 根据硬件选择
```
GTX 1060/RTX 3060 → eager
RTX 3070/RTX 4060 → sdpa  
RTX 3080以上 → flash_attention_2
```

### 根据任务选择
```
快速测试 → flash_attention_2
平衡性能 → sdpa
稳定性优先 → eager
```

## 性能对比

| 机制 | 速度 | 内存 | 推荐度 |
|------|------|------|--------|
| eager | ⭐⭐ | ⭐⭐⭐ | 备用 |
| sdpa | ⭐⭐⭐ | ⭐⭐⭐ | 推荐 |
| flash_attention_2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 最优 |

## 代码示例

```python
# 在节点中的使用
attention="flash_attention_2"  # 最佳性能
attention="sdpa"               # 平衡选择  
attention="eager"              # 兼容性最佳
```

## 注意事项

1. **flash_attention_2** 需要较新的GPU和驱动支持
2. 如果出现兼容性问题，降级到 **sdpa**
3. 在老设备上，**eager** 可能更稳定
4. 我们的节点默认设置为 **flash_attention_2** 以获得最佳性能