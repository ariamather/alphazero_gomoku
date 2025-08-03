# trainer_resume.py
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
        
        # 如果配置中指定了resume_checkpoint，则根据resume_mode加载
        if 'resume_checkpoint' in config and config['resume_checkpoint']:
            resume_mode = config.get('resume_mode', 'weights_only')
            self.load_from_checkpoint(config['resume_checkpoint'], mode=resume_mode)
        
        # 创建新的保存目录
        self.save_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if 'resume_checkpoint' in config and config['resume_checkpoint']:
            # 在目录名中标记这是从哪个模型恢复的
            checkpoint_name = os.path.basename(config['resume_checkpoint']).split('.')[0]
            self.save_dir = f"{self.save_dir}_from_{checkpoint_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存本次训练的配置
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

    def load_from_checkpoint(self, checkpoint_path, mode='weights_only'):
        """
        从检查点加载状态
        Args:
            checkpoint_path (str): 检查点文件路径
            mode (str): 加载模式
                - 'weights_only': 只加载模型权重
                - 'weights_optimizer': 加载模型权重和优化器状态
                - 'full': 加载所有状态（模型、优化器、调度器、统计信息）
        """
        print(f"Loading checkpoint from {checkpoint_path} with mode '{mode}'...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device ,weights_only = False)
        
        # 1. 总是加载模型权重
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded")
        
        # 2. 根据模式加载其他组件
        if mode in ['weights_optimizer', 'full']:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state loaded")
            else:
                print("⚠ Optimizer state not found in checkpoint")
        
        if mode == 'full':
            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state loaded")
            else:
                print("⚠ Scheduler state not found in checkpoint")
            
            # 加载统计信息
            if 'stats' in checkpoint:
                self.stats = checkpoint['stats']
                print("✓ Training statistics loaded")
                # 显示之前的训练进度
                if self.stats['iteration']:
                    last_iter = self.stats['iteration'][-1]
                    print(f"  Previous training stopped at iteration {last_iter}")
                    if self.stats['win_rate']:
                        last_win_rate = self.stats['win_rate'][-1]['win_rate']
                        print(f"  Last evaluation win rate: {last_win_rate:.2%}")
            else:
                print("⚠ Training statistics not found in checkpoint")
            
            # 可选：加载回放缓冲区
            if self.config.get('load_replay_buffer', False) and 'replay_buffer' in checkpoint:
                self.replay_buffer.extend(checkpoint['replay_buffer'])
                print(f"✓ Replay buffer loaded ({len(self.replay_buffer)} samples)")
        
        # 显示加载的配置信息（如果有）
        if 'config' in checkpoint:
            old_config = checkpoint['config']
            print("\nPrevious training configuration:")
            print(f"  Board size: {old_config.get('board_size', 'N/A')}")
            print(f"  Learning rate: {old_config.get('learning_rate', 'N/A')}")
            print(f"  Batch size: {old_config.get('batch_size', 'N/A')}")
            
            # 检查关键配置是否匹配
            if old_config.get('board_size') != self.config['board_size']:
                print(f"\n⚠ WARNING: Board size mismatch! Old: {old_config.get('board_size')}, New: {self.config['board_size']}")
            if old_config.get('num_res_blocks') != self.config['num_res_blocks']:
                print(f"⚠ WARNING: Network architecture mismatch! This may cause errors.")
        
        print(f"\nCheckpoint loaded successfully in '{mode}' mode!")

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
            'config': self.config,
            'iteration': iteration
        }
        
        # 可选：保存回放缓冲区
        if self.config.get('save_replay_buffer', False):
            checkpoint['replay_buffer'] = list(self.replay_buffer)
        
        torch.save(checkpoint, path)
        with open(f"{self.save_dir}/stats.json", 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        """向后兼容的加载方法"""
        checkpoint = torch.load(path, map_location=self.device , weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.stats = checkpoint['stats']
        print(f"Checkpoint loaded from {path}")

    def run(self):
        """主训练循环"""
        # 显示训练开始信息
        if 'resume_checkpoint' in self.config and self.config['resume_checkpoint']:
            print(f"\n{'='*50}")
            print(f"Starting new training from checkpoint: {self.config['resume_checkpoint']}")
            print(f"Resume mode: {self.config.get('resume_mode', 'weights_only')}")
            print(f"{'='*50}\n")
        
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
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")
            
            # 3. 评估与保存
            if iteration % self.config['eval_interval'] == 0:
                win_rate = self.evaluate()
                self.stats['win_rate'].append({'iteration': iteration, 'win_rate': win_rate})
            
            if iteration % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(iteration)
        
        self.save_checkpoint('final')
        print("\nTraining completed!")