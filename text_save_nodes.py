import json
import os
from pathlib import Path
import folder_paths

class Qwen3_Text_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file_path": ("STRING", {"default": "", "placeholder": "Enter file path with extension (e.g., output.txt or output.xlsx)\nRelative paths will use ComfyUI output directory as base"}),
                "file_format": ("STRING", {"default": "txt", "choices": ["txt", "excel"], "forceInput": False}),
                "mode": ("STRING", {"default": "append", "choices": ["append", "write"]}),
            },
            "optional": {
                "sheet_name": ("STRING", {"default": "Sheet1"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("status", "file_path",)
    FUNCTION = "save_text"
    CATEGORY = "Qwen3-VL Utils"

    def is_json_string(self, text):
        """检查文本是否为JSON字符串"""
        try:
            json.loads(text)
            return True
        except ValueError:
            return False

    def format_json_to_line(self, text):
        """将JSON字符串格式化为单行"""
        try:
            data = json.loads(text)
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except ValueError:
            return text

    def save_to_txt(self, file_path, text, mode):
        """保存文本到txt文件"""
        try:
            # 确保父目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 处理JSON格式
            if self.is_json_string(text):
                text = self.format_json_to_line(text)
            else:
                # 将普通文本转换为单行，替换所有换行符和回车符为空格
                text = text.replace('\n', ' ').replace('\r', '').strip()
            
            # 将用户选择的模式映射为Python实际模式字符串
            mode_map = {
                "write": "w",
                "append": "a"
            }
            actual_mode = mode_map.get(mode, "a") + 't'  # 默认使用追加模式
            
            # 写入文件
            with open(file_path, actual_mode, encoding='utf-8') as f:
                f.write(text + '\n')
            
            return "Success: Text saved to txt file", file_path
        except Exception as e:
            return f"Error: {str(e)}", file_path

    def save_to_excel(self, file_path, text, mode, sheet_name):
        """保存文本到excel文件"""
        try:
            # 确保父目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 处理JSON格式
            if self.is_json_string(text):
                text = self.format_json_to_line(text)
            else:
                # 将普通文本转换为单行，替换所有换行符和回车符为空格
                text = text.replace('\n', ' ').replace('\r', '').strip()
            
            # 使用openpyxl操作excel
            from openpyxl import Workbook, load_workbook
            
            try:
                # 尝试加载现有文件
                wb = load_workbook(file_path)
                if sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                else:
                    # 工作表不存在则创建
                    ws = wb.create_sheet(title=sheet_name)
                    wb.active = ws
            except Exception as e:
                # 任何加载错误都创建新文件
                print(f"创建新Excel文件: {e}")
                wb = Workbook()
                ws = wb.active
                ws.title = sheet_name
            
            if mode == "write":
                # 清空工作表内容
                if ws.max_row > 1:
                    ws.delete_rows(2, ws.max_row)
                if ws.max_row >= 1:
                    ws.delete_rows(1)
                ws.append([text])
            else:
                # 追加到工作表末尾
                ws.append([text])
            
            # 保存文件
            wb.save(file_path)
            wb.close()
            
            return "Success: Text saved to excel file", file_path
        except Exception as e:
            return f"Error: {str(e)}", file_path

    def process_file_path(self, file_path):
        """处理文件路径，如果是相对路径则使用ComfyUI的output目录作为基础"""
        if not file_path:
            return ""
        
        # 获取ComfyUI的output目录
        output_dir = folder_paths.get_output_directory()
        
        # Windows和Unix风格的绝对路径判断
        is_absolute = os.path.isabs(file_path)
        
        # Windows特殊处理：以斜杠开头的路径（如/文件夹/文件）在Windows中被视为绝对路径
        # 但用户可能希望这是相对路径，所以特殊处理
        if file_path.startswith('/') or file_path.startswith('\\'):
            # 以斜杠开头的路径，视为相对路径
            is_absolute = False
        
        if is_absolute:
            # 绝对路径，直接返回
            return file_path
        else:
            # 相对路径，拼接output目录
            # 清理路径中的..和.以避免路径遍历安全问题
            # 如果原始路径以/或\开头，需要先去除这个前缀
            if file_path.startswith('/') or file_path.startswith('\\'):
                # 去除开头的斜杠
                safe_path = file_path[1:].lstrip('/\\')
            else:
                safe_path = file_path
            
            # 进一步清理路径中的.和..
            safe_path = os.path.normpath(safe_path)
            # 确保不在output目录之外
            full_path = os.path.join(output_dir, safe_path)
            return full_path

    def save_text(self, text, file_path, file_format, mode, sheet_name="Sheet1"):
        """主保存函数"""
        # 验证文件路径
        if not file_path:
            return "Error: File path is empty", ""
        
        # 处理路径
        processed_path = self.process_file_path(file_path)
        if not processed_path:
            return "Error: Failed to process file path", ""
        
        # 根据文件格式选择保存方法
        if file_format == "txt":
            return self.save_to_txt(processed_path, text, mode)
        elif file_format == "excel":
            return self.save_to_excel(processed_path, text, mode, sheet_name)
        else:
            return "Error: Unsupported file format", file_path