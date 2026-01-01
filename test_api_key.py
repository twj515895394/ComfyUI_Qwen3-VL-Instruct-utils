import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen3_vl_api_nodes import get_api_key

try:
    api_key = get_api_key()
    print(f"API key: {api_key}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")