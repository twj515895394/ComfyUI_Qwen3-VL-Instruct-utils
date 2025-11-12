from .nodes import Qwen3_VQA, Qwen3_VQA_Quick
from .util_nodes import ImageLoader, VideoLoader, VideoLoaderPath
from .path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen3 VQA": Qwen3_VQA,
    "Qwen3 VQA Quick": Qwen3_VQA_Quick,
    "Load Image Advanced": ImageLoader,
    "Load Video Advanced": VideoLoader,
    "Load Video Advanced (Path)": VideoLoaderPath,
    "Multiple Paths Input": MultiplePathsInput,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3 VQA": "Qwen3 VQA",
    "Qwen3 VQA Quick": "Qwen3 VQA Quick",
    "Load Image Advanced": "Load Image Advanced",
    "Load Video Advanced": "Load Video Advanced",
    "Load Video Advanced (Path)": "Load Video Advanced (Path)",
    "Multiple Paths Input": "Multiple Paths Input",
}
