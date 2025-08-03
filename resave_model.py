import torch
import numpy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
output_path = os.path.join(os.path.dirname(__file__), 'best_simple.pt')

# 加载原始模型，使用安全全局变量上下文管理器
with torch.serialization.safe_globals([numpy]):
    checkpoint = torch.load(model_path, map_location=device)

# 提取模型状态字典
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
else:
    model_state_dict = checkpoint

# 只保存模型状态字典
torch.save(model_state_dict, output_path, _use_new_zipfile_serialization=False)

print(f"Model has been resaved successfully to {output_path}")