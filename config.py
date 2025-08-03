# config.py
import torch

# 训练配置
CONFIG = {
    # 训练流程控制
    'num_iterations': 500,
    'num_epochs': 10,
    'checkpoint_interval': 20,
    'eval_interval': 10,
    
    # 自我对弈 (Self-Play)
    'num_self_play_games': 72,
    'replay_buffer_size': 150000,
    'temperature_threshold': 10, # 多少步之后温度变为0

    # MCTS
    'num_mcts_simulations': 1200,
    'c_puct': 1.0,
    'dirichlet_alpha': 0.3,
    'noise_weight': 0.25,

    # 神经网络与优化器 (Network & Optimizer)
    'num_res_blocks': 12,
    'num_channels': 256,
    'batch_size': 2048,
    'learning_rate': 2e-5,
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,

    # 评估 (Evaluation)
    'eval_games': 10,
    'eval_opponent': 'random', # 可以是 'random' 或其他

    # 环境与设备 (Environment & Device)
    'board_size': 12,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 恢复训练设置
    'resume_checkpoint': '/home/chenhx/gobangAI/gomoku_alpha_zero/checkpoints/20250801_013029/checkpoint_300.pt',
    'resume_mode': 'weights_optimizer'  # 可选：'weights_only', 'weights_optimizer', 'full'
}