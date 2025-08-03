# trainer.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import json
from datetime import datetime
from tqdm import tqdm

from network import AlphaZeroNet
from self_play import run_self_play, Transition
from GomokuGame import GomokuEnv, Player
from mcts import MCTS

class AlphaZeroTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.board_size = config['board_size']
        
        self.net = AlphaZeroNet(
            self.board_size, 
            config['num_res_blocks'],
            config['num_channels'],
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_iterations']
        )
        
        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        self.stats = {'iteration': [], 'loss': [], 'value_loss': [], 'policy_loss': [], 'win_rate': []}
        
        self.save_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        # 保存本次训练的配置
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

    def _train_step(self, batch):
        obs_batch = torch.tensor(np.array([t.obs for t in batch]), dtype=torch.float32).to(self.device)
        pi_batch = torch.tensor(np.array([t.pi for t in batch]), dtype=torch.float32).to(self.device)
        z_batch = torch.tensor(np.array([t.z for t in batch]), dtype=torch.float32).to(self.device)
        
        self.net.train()
        logits, values = self.net(obs_batch)
        
        value_loss = F.mse_loss(values, z_batch)
        policy_loss = -torch.mean(torch.sum(pi_batch * F.log_softmax(logits, dim=1), dim=1))
        total_loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.config['grad_clip_norm'])
        self.optimizer.step()
        
        return {'loss': total_loss.item(), 'value_loss': value_loss.item(), 'policy_loss': policy_loss.item()}

    def evaluate(self):
        """评估当前网络对战随机对手的胜率"""
        print(f"Evaluating against '{self.config['eval_opponent']}'...")
        wins, draws, losses = 0, 0, 0
        self.net.eval()
        device = torch.device(self.config['device'])
        mcts = MCTS(self.net, self.config, device)


        for _ in tqdm(range(self.config['eval_games']), desc="Evaluation"):
            env = GomokuEnv(self.board_size)
            obs, info = env.reset()
            ai_player = random.choice([Player.BLACK, Player.WHITE])
            
            while True:
                if env.current_player == ai_player:
                    visits = mcts.run(env, obs, add_noise=False) # 评估时不加噪声
                    action = np.argmax(visits)
                else: # 对手
                    valid_actions = env.get_valid_actions()
                    if len(valid_actions)==0: break
                    action = random.choice(valid_actions)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    if env.winner == ai_player:
                        wins += 1
                    elif env.winner is None:
                        draws += 1
                    else:
                        losses += 1
                    break
        
        win_rate = wins / self.config['eval_games'] if self.config['eval_games'] > 0 else 0
        print(f"Eval Results - Win Rate: {win_rate:.2%} (Wins: {wins}, Draws: {draws}, Losses: {losses})")
        return win_rate

    def save_checkpoint(self, iteration):
        path = f"{self.save_dir}/checkpoint_{iteration}.pt"
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stats': self.stats,
            'config': self.config
        }
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
        with open(f"{self.save_dir}/stats.json", 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.stats = checkpoint['stats']
        print(f"Checkpoint loaded from {path}")

    def run(self):
        """主训练循环"""
        for iteration in range(1, self.config['num_iterations'] + 1):
            print(f"\n{'='*20} Iteration {iteration}/{self.config['num_iterations']} {'='*20}")
            
            # 1. 自我对弈生成数据
            self.net.eval() # 自我对弈时，模型处于评估模式
            data = run_self_play(self.net, self.config)
            self.replay_buffer.extend(data)
            print(f"Self-play finished. Generated {len(data)} transitions. Buffer size: {len(self.replay_buffer)}")
            
            # 2. 训练网络
            if len(self.replay_buffer) < self.config['batch_size']:
                print("Replay buffer too small, skipping training for this iteration.")
                continue

            epoch_losses = []
            for epoch in tqdm(range(self.config['num_epochs']), desc="Training epochs"):
                batch = random.sample(self.replay_buffer, self.config['batch_size'])
                losses = self._train_step(batch)
                epoch_losses.append(losses)
            # 计算平均loss
            losses = {k: np.mean([l[k] for l in epoch_losses]) for k in losses.keys()}
            
            print(f"Avg Loss: {losses['loss']:.4f} | Value: {losses['value_loss']:.4f} | Policy: {losses['policy_loss']:.4f}")
            self.stats['iteration'].append(iteration)
            self.stats['loss'].append(losses['loss'])
            self.stats['value_loss'].append(losses['value_loss'])
            self.stats['policy_loss'].append(losses['policy_loss'])
            
            self.scheduler.step()
            
            # 3. 评估与保存
            if iteration % self.config['eval_interval'] == 0:
                win_rate = self.evaluate()
                self.stats['win_rate'].append({'iteration': iteration, 'win_rate': win_rate})
            
            if iteration % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(iteration)
        
        self.save_checkpoint('final')
        print("\nTraining completed!")