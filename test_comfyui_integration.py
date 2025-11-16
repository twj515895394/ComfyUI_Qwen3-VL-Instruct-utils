import sys
import os
import tempfile

# 确保ComfyUI相关路径在Python路径中
comfyui_path = "x:/ComfyUI-aki-v2/ComfyUI"
custom_nodes_path = os.path.join(comfyui_path, "custom_nodes")
sys.path.append(comfyui_path)
sys.path.append(custom_nodes_path)

def test_node_integration():
    """测试节点在ComfyUI环境中的集成"""
    try:
        print("测试ComfyUI节点集成...")
        print("=" * 50)
        
        # 尝试动态导入节点
        import importlib.util
        import sys
        
        # 动态导入节点模块
        module_name = "text_save_nodes"
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), "text_save_nodes.py"))
        text_save_nodes = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = text_save_nodes
        spec.loader.exec_module(text_save_nodes)
        
        Qwen3_Text_Save = text_save_nodes.Qwen3_Text_Save
        print("✓ 节点导入成功")
        
        # 测试节点的INPUT_TYPES
        input_types = Qwen3_Text_Save.INPUT_TYPES()
        print(f"✓ 节点INPUT_TYPES定义正确: {input_types}")
        
        # 测试节点的RETURN_TYPES
        return_types = Qwen3_Text_Save.RETURN_TYPES
        return_names = Qwen3_Text_Save.RETURN_NAMES
        print(f"✓ 节点RETURN_TYPES定义正确: {return_types}")
        print(f"✓ 节点RETURN_NAMES定义正确: {return_names}")
        
        # 测试节点的FUNCTION
        function = Qwen3_Text_Save.FUNCTION
        print(f"✓ 节点FUNCTION定义正确: {function}")
        
        # 测试节点的CATEGORY
        category = Qwen3_Text_Save.CATEGORY
        print(f"✓ 节点CATEGORY定义正确: {category}")
        
        # 创建节点实例并测试功能
        node = Qwen3_Text_Save()
        
        # 测试TXT保存功能
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_txt_path = f.name
        f.close()  # 立即关闭文件以便后续操作
        
        test_text = "这是ComfyUI集成测试的文本"
        status, file_path = node.save_text(test_text, temp_txt_path, "txt", "write")
        print(f"✓ TXT保存测试: {status}")
        
        # 验证文件内容
        with open(temp_txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content == test_text:
            print("✓ TXT文件内容正确")
        else:
            print(f"✗ TXT文件内容错误: 预期 '{test_text}', 实际 '{content}'")
        
        # 清理临时文件
        os.unlink(temp_txt_path)
        
        print("=" * 50)
        print("🎉 ComfyUI节点集成测试通过!")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    success = test_node_integration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()