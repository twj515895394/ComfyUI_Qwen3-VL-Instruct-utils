import os
import json
import openpyxl
from openpyxl.utils import get_column_letter

class Qwen3_Text_Excel_Reader:
    """
    读取文本或 Excel 文件的节点
    支持设置起始行、读取行数，输出文本批次集合和总文本行数
    """
    NAME = "Qwen3 Text/Excel Reader"
    CATEGORY = "Qwen3-VL-Instruct"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "文件路径 (.txt/.xlsx)"}),
                "file_format": ("STRING", {"default": "text", "choices": ["text", "excel"]}),
                "start_line": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "read_count": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "excel_sheet": ("STRING", {"default": "Sheet1"}),
            }
        }

    RETURN_TYPES = ("BATCH_TEXT", "INT")
    RETURN_NAMES = ("text_batches", "text_line_count")
    FUNCTION = "read_file"

    def read_file(self, file_path, file_format, start_line, read_count, excel_sheet):
        """
        读取文件内容并输出文本批次
        :param file_path: 文件路径
        :param file_format: 文件格式 (text/excel)
        :param start_line: 起始行号 (从 1 开始)
        :param read_count: 读取行数 (-1 表示全部)
        :param excel_sheet: Excel 工作表名称
        :return: (text_batches, text_line_count)
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")

        text_lines = []

        if file_format == "text":
            # 读取文本文件
            with open(file_path, "r", encoding="utf-8") as f:
                # 跳过前 start_line-1 行
                for _ in range(start_line - 1):
                    next(f, None)

                # 读取指定行数
                if read_count > 0:
                    for _ in range(read_count):
                        line = f.readline().strip()
                        if not line:
                            continue  # 跳过空行
                        text_lines.append(line)
                else:
                    # 读取剩余所有行
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue  # 跳过空行
                        text_lines.append(line)
        else:  # excel
            # 读取 Excel 文件
            wb = openpyxl.load_workbook(file_path, read_only=True)
            sheet = wb[excel_sheet] if excel_sheet in wb.sheetnames else wb.active

            # 跳过前 start_line-1 行
            rows = sheet.iter_rows(min_row=start_line, values_only=True)

            # 读取指定行数
            if read_count > 0:
                for _ in range(read_count):
                    row = next(rows, None)
                    if row is None:
                        break
                    # 将一行转换为字符串 (使用 TAB 分隔列)
                    line = "\t".join([str(cell) if cell is not None else "" for cell in row])
                    if not line:
                        continue  # 跳过空行
                    text_lines.append(line)
            else:
                # 读取剩余所有行
                for row in rows:
                    if row is None:
                        break
                    line = "\t".join([str(cell) if cell is not None else "" for cell in row])
                    if not line:
                        continue  # 跳过空行
                    text_lines.append(line)

            wb.close()

        # 将文本行转换为批次格式
        text_batches = []
        for line in text_lines:
            text_batches.append(line)

        return (tuple(text_batches), len(text_batches))


class Qwen3_Text_Batch_Loader:
    """
    文本批次读取节点
    类似图像批次读取，支持设置读取索引，输出文本内容
    """
    NAME = "Qwen3 Text Batch Loader"
    CATEGORY = "Qwen3-VL-Instruct"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_batches": ("BATCH_TEXT",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_content",)
    FUNCTION = "load_batch"

    def load_batch(self, text_batches, batch_index):
        """
        从文本批次中读取指定索引的文本内容
        :param text_batches: 文本批次集合
        :param batch_index: 读取索引 (从 0 开始)
        :return: text_content
        """
        if not text_batches:
            raise ValueError("文本批次为空")

        if batch_index < 0 or batch_index >= len(text_batches):
            raise ValueError(f"索引超出范围: {batch_index}, 总批次: {len(text_batches)}")

        return (text_batches[batch_index],)


# 定义自定义节点类型
NODE_CLASS_MAPPINGS = {
    "Qwen3_Text_Excel_Reader": Qwen3_Text_Excel_Reader,
    "Qwen3_Text_Batch_Loader": Qwen3_Text_Batch_Loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_Text_Excel_Reader": "Qwen3 Text/Excel Reader",
    "Qwen3_Text_Batch_Loader": "Qwen3 Text Batch Loader"
}