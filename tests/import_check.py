import asyncio
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, r'E:/AI_painting/ComfyUI')
root = Path(r'E:/AI_painting/ComfyUI/custom_nodes/ComfyUI-SelfLift-Split')
name = 'custom_nodes_ComfyUI_SelfLift_Split'
spec = importlib.util.spec_from_file_location(name, root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
print(sorted(module.NODE_CLASS_MAPPINGS))
assert asyncio.run(__import__('nodes').load_custom_node(str(root)))
import nodes
print(all(k in nodes.NODE_CLASS_MAPPINGS for k in module.NODE_CLASS_MAPPINGS))
