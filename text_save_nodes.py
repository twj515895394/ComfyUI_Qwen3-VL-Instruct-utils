import json
import os
from pathlib import Path

class Qwen3_Text_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file_path": ("STRING", {"default": "", "placeholder": "Enter file path with extension (e.g., output.txt or output.xlsx)"}),
                "file_format": ("STRING", {"default": "txt", "choices": ["txt", "excel"]}),
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

    def save_text(self, text, file_path, file_format, mode, sheet_name="Sheet1"):
        """主保存函数"""
        # 验证文件路径
        if not file_path:
            return "Error: File path is empty", ""
        
        # 根据文件格式选择保存方法
        if file_format == "txt":
            return self.save_to_txt(file_path, text, mode)
        elif file_format == "excel":
            return self.save_to_excel(file_path, text, mode, sheet_name)
        else:
            return "Error: Unsupported file format", file_path