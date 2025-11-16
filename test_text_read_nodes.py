import os
import json
import tempfile
import openpyxl
from text_read_nodes import Qwen3_Text_Excel_Reader, Qwen3_Text_Batch_Loader

def test_text_reader_node():
    """
    测试文本文件读取节点
    """
    print("=== 测试文本文件读取节点 ===")
    
    try:
        # 创建临时文本文件
        test_content = "第1行内容\n第2行内容\n第3行内容\n第4行内容\n第5行内容"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file_path = f.name
        
        # 测试节点
        node = Qwen3_Text_Excel_Reader()
        
        # 测试1: 读取全部内容
        print("测试1: 读取全部内容")
        batches, count = node.read_file(temp_file_path, "text", 1, -1, "Sheet1")
        print(f"总行数: {count}")
        print(f"文本批次: {batches}")
        assert count == 5, f"预期读取5行，实际读取{count}行"
        assert len(batches) == 5, f"预期批次长度5，实际{len(batches)}"
        print("✓ 测试1通过")
        
        # 测试2: 从第2行开始读取3行
        print("\n测试2: 从第2行开始读取3行")
        batches, count = node.read_file(temp_file_path, "text", 2, 3, "Sheet1")
        print(f"总行数: {count}")
        print(f"文本批次: {batches}")
        assert count == 3, f"预期读取3行，实际读取{count}行"
        assert batches[0] == "第2行内容", f"预期第1个批次为'第2行内容'，实际{batches[0]}"
        assert batches[-1] == "第4行内容", f"预期第3个批次为'第4行内容'，实际{batches[-1]}"
        print("✓ 测试2通过")
        
        # 测试3: 跳过空行
        print("\n测试3: 读取包含空行的文件")
        test_content_with_empty = "第1行\n\n第3行\n\n第5行"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content_with_empty)
        batches, count = node.read_file(temp_file_path, "text", 1, -1, "Sheet1")
        print(f"总行数: {count}")
        print(f"文本批次: {batches}")
        assert count == 3, f"预期读取3行(跳过空行)，实际读取{count}行"
        print("✓ 测试3通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                # 如果文件被占用，暂时跳过清理
                pass
    
    print("\n=== 文本文件读取节点测试全部通过 ===")
    return True

def test_excel_reader_node():
    """
    测试Excel文件读取节点
    """
    print("\n=== 测试Excel文件读取节点 ===")
    
    try:
        # 创建临时Excel文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_file_path = f.name
        
        # 写入测试数据
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "TestSheet"
        sheet.append(["Column1", "Column2", "Column3"])
        sheet.append(["行1列1", "行1列2", "行1列3"])
        sheet.append(["行2列1", "行2列2", "行2列3"])
        sheet.append(["行3列1", "行3列2", "行3列3"])
        wb.save(temp_file_path)
        wb.close()
        
        # 测试节点
        node = Qwen3_Text_Excel_Reader()
        
        # 测试1: 读取全部内容
        print("测试1: 读取全部内容")
        batches, count = node.read_file(temp_file_path, "excel", 1, -1, "TestSheet")
        print(f"总行数: {count}")
        print(f"文本批次: {batches}")
        assert count == 4, f"预期读取4行，实际读取{count}行"
        print("✓ 测试1通过")
        
        # 测试2: 从第2行开始读取2行
        print("\n测试2: 从第2行开始读取2行")
        batches, count = node.read_file(temp_file_path, "excel", 2, 2, "TestSheet")
        print(f"总行数: {count}")
        print(f"文本批次: {batches}")
        assert count == 2, f"预期读取2行，实际读取{count}行"
        assert "行1列1\t行1列2\t行1列3" in batches[0], f"预期包含行1数据"
        assert "行2列1\t行2列2\t行2列3" in batches[1], f"预期包含行2数据"
        print("✓ 测试2通过")
        
        # 测试3: 使用默认工作表名称
        print("\n测试3: 使用默认工作表名称")
        batches, count = node.read_file(temp_file_path, "excel", 1, 1, "Sheet1")
        print(f"总行数: {count}")
        print("✓ 测试3通过")
        
    except ImportError:
        print("⚠ Excel测试跳过: openpyxl未安装")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                # 如果文件被占用，暂时跳过清理
                pass
    
    print("\n=== Excel文件读取节点测试全部通过 ===")
    return True

def test_batch_loader_node():
    """
    测试文本批次读取节点
    """
    print("\n=== 测试文本批次读取节点 ===")
    
    try:
        # 创建测试批次
        test_batches = ("文本1", "文本2", "文本3", "文本4", "文本5")
        
        # 测试节点
        node = Qwen3_Text_Batch_Loader()
        
        # 测试1: 读取索引0
        print("测试1: 读取索引0")
        text = node.load_batch(test_batches, 0)
        print(f"读取内容: {text}")
        assert text[0] == "文本1", f"预期读取'文本1'，实际读取{text[0]}"
        print("✓ 测试1通过")
        
        # 测试2: 读取索引2
        print("\n测试2: 读取索引2")
        text = node.load_batch(test_batches, 2)
        print(f"读取内容: {text}")
        assert text[0] == "文本3", f"预期读取'文本3'，实际读取{text[0]}"
        print("✓ 测试2通过")
        
        # 测试3: 读取最后一个索引
        print("\n测试3: 读取最后一个索引")
        text = node.load_batch(test_batches, 4)
        print(f"读取内容: {text}")
        assert text[0] == "文本5", f"预期读取'文本5'，实际读取{text[0]}"
        print("✓ 测试3通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    print("\n=== 文本批次读取节点测试全部通过 ===")
    return True

def main():
    """
    运行所有测试
    """
    print("开始测试文本/Excel读取节点")
    print("=" * 50)
    
    test_results = []
    test_results.append("文本文件读取节点: " + ("✓ 通过" if test_text_reader_node() else "✗ 失败"))
    test_results.append("Excel文件读取节点: " + ("✓ 通过" if test_excel_reader_node() else "✗ 失败"))
    test_results.append("文本批次读取节点: " + ("✓ 通过" if test_batch_loader_node() else "✗ 失败"))
    
    print("=" * 50)
    print("测试总结:")
    for result in test_results:
        print(result)
    
    print("=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()