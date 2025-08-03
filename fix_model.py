import torch
import numpy
import os
from network import AlphaZeroNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
output_path = os.path.join(os.path.dirname(__file__), 'best_fixed.pt')

# 尝试多种方式获取numpy的scalar类型
try:
    # 方式1: 使用numpy._core.multiarray.scalar
    numpy_scalar = numpy._core.multiarray.scalar
    print("Successfully obtained numpy._core.multiarray.scalar")
except AttributeError:
    try:
        # 方式2: 使用numpy.core.multiarray.scalar (不带下划线)
        numpy_scalar = numpy.core.multiarray.scalar
        print("Successfully obtained numpy.core.multiarray.scalar")
    except AttributeError:
        try:
            # 方式3: 直接导入numpy.core.multiarray
            from numpy.core import multiarray
            numpy_scalar = multiarray.scalar
            print("Successfully obtained multiarray.scalar")
        except (AttributeError, ImportError):
            # 退回到使用numpy.generic
            numpy_scalar = numpy.generic
            print("Using numpy.generic instead")

# 最后尝试: 不使用安全全局变量，直接加载并保存模型
try:
    # 尝试不使用安全全局变量加载
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("Model loaded successfully without safe_globals")

    # 提取模型状态字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    # 保存干净的模型状态字典
    torch.save(model_state_dict, output_path, _use_new_zipfile_serialization=False)
    print(f"Model saved successfully to {output_path}")
except Exception as e:
    print(f"Error processing model: {e}")
    raise

# 如果以上都失败，尝试使用pickle模块加载
import pickle
try:
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print("Model loaded successfully with pickle")

    # 提取模型状态字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    # 保存干净的模型状态字典
    torch.save(model_state_dict, output_path, _use_new_zipfile_serialization=False)
    print(f"Model saved successfully to {output_path}")
except Exception as e:
    print(f"Error processing model with pickle: {e}")
    raise