# utils.py
import numpy as np

def augment_data(obs, pi, board_size):
    """数据增强：旋转和翻转"""
    augmented_data = []
    
    # 将pi重塑为2D
    pi_2d = pi.reshape(board_size, board_size)
    
    for k in range(4):  # 4个旋转角度
        # 旋转
        obs_rot = np.rot90(obs, k, axes=(1, 2)).copy() # .copy() 避免内存问题
        pi_rot = np.rot90(pi_2d, k).copy()
        augmented_data.append((obs_rot, pi_rot.flatten()))
        
        # 水平翻转
        obs_flip = np.flip(obs_rot, axis=2).copy()
        pi_flip = np.flip(pi_rot, axis=1).copy()
        augmented_data.append((obs_flip, pi_flip.flatten()))
    
    return augmented_data