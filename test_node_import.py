#!/usr/bin/env python3
"""
测试节点是否能正确导入
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试导入节点...")
    
    # 测试导入文本保存节点
    from text_save_nodes import Qwen3_Text_Save
    print("✓ 成功导入 Qwen3_Text_Save")
    
    # 测试导入文本读取节点
    from text_read_nodes import Qwen3_Text_Excel_Reader, Qwen3_Text_Batch_Loader
    print("✓ 成功导入 Qwen3_Text_Excel_Reader")
    print("✓ 成功导入 Qwen3_Text_Batch_Loader")
    
    # 测试节点类是否有INPUT_TYPES属性
    assert hasattr(Qwen3_Text_Save, 'INPUT_TYPES'), "Qwen3_Text_Save 缺少 INPUT_TYPES"
    assert hasattr(Qwen3_Text_Excel_Reader, 'INPUT_TYPES'), "Qwen3_Text_Excel_Reader 缺少 INPUT_TYPES"
    assert hasattr(Qwen3_Text_Batch_Loader, 'INPUT_TYPES'), "Qwen3_Text_Batch_Loader 缺少 INPUT_TYPES"
    print("✓ 所有节点都有 INPUT_TYPES 属性")
    
    # 测试节点类是否有FUNCTION属性
    assert hasattr(Qwen3_Text_Save, 'FUNCTION'), "Qwen3_Text_Save 缺少 FUNCTION"
    assert hasattr(Qwen3_Text_Excel_Reader, 'FUNCTION'), "Qwen3_Text_Excel_Reader 缺少 FUNCTION"
    assert hasattr(Qwen3_Text_Batch_Loader, 'FUNCTION'), "Qwen3_Text_Batch_Loader 缺少 FUNCTION"
    print("✓ 所有节点都有 FUNCTION 属性")
    
    print("\n所有测试通过！节点导入成功！")
    
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)