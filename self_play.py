# self_play.py
import numpy as np
import torch
import os
from collections import namedtuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 假设GomokuEnv和Player在GomokuGame.py中
from GomokuGame import GomokuEnv
from mcts import MCTS
from utils import augment_data
from network import AlphaZeroNet

Transition = namedtuple('Transition', ['obs', 'pi', 'z'])

def _self_play_game(net, config,device):
    """进行一局自我对弈 (内部函数)"""
    board_size = config['board_size']
    env = GomokuEnv(board_size)
    obs, info = env.reset()
    pis=[]
    
    states, mcts_visits, current_players = [], [], []
    move_count = 0
    mcts = MCTS(net, config,device)
    
    while True:
        temperature = 1.0 if move_count < config['temperature_threshold'] else 0.0
        
        visits = mcts.run(env, obs)
        
        # 确保visits是有效的
        assert visits.sum() > 0, "MCTS visits sum is 0!"
        
        # 计算概率分布
        if temperature > 0:
            visits_temp = np.power(visits, 1.0 / temperature)
            pi = visits_temp / np.sum(visits_temp)
        else:
            pi = np.zeros_like(visits)
            pi[np.argmax(visits)] = 1.0

        states.append(obs.copy())
        mcts_visits.append(visits)
        current_players.append(env.current_player.value)
        pis.append(pi)
        
        action = np.random.choice(len(pi), p=pi)
        
        obs, reward, terminated, truncated, info = env.step(action)
        move_count += 1
        
        if terminated or truncated:
            break
    
    # 确定游戏结果
    if env.winner is not None:
        winner_value = env.winner.value
    else:
        winner_value = 0
    
    data = []
    normalized_pis = [visits / np.sum(visits) for visits in mcts_visits]

    for state, norm_pi, player in zip(states, pis, current_players):
        # 正确分配价值
        if winner_value == 0:
            z = 0.0
        else:
            z = 1.0 if player == winner_value else -1.0
        
        augmented = augment_data(state, norm_pi, board_size)
        for aug_state, aug_pi in augmented:
            data.append(Transition(aug_state, aug_pi, z))
    
    return data

def _worker(args_tuple):
    """多进程工作函数"""
    game_id, net_state_path, config, gpu_id = args_tuple  # 解包gpu_id
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    net_copy = AlphaZeroNet(
        config['board_size'], 
        config['num_res_blocks'], 
        config['num_channels']
    ).to(device)
    net_copy.load_state_dict(torch.load(net_state_path, map_location=device))
    net_copy.eval()
    
    return _self_play_game(net_copy, config,device)

def run_self_play(net, config):
    """并行运行自我对弈"""
    num_gpus = torch.cuda.device_count()
    num_games = config['num_self_play_games']
    data = []
    
    temp_net_path = "temp_net_for_selfplay.pt"
    torch.save(net.state_dict(), temp_net_path)
    
    num_processes = max(1, cpu_count()) # 占用所有CPU
    print(f"Using {num_processes} processes for self-play...")
    
    args_list = [(i, temp_net_path, config, i % num_gpus) for i in range(num_games)]  # 添加gpu_id

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_worker, args_list), total=num_games, desc="Self-play"))
    
    for game_data in results:
        data.extend(game_data)
    
    if os.path.exists(temp_net_path):
        os.remove(temp_net_path)
    
    return data