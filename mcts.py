# mcts.py
import numpy as np
import torch
import torch.nn.functional as F
import copy

class MCTSNode:
    def __init__(self, parent, prior, action):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.action = action
        self.is_expanded = False
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct):
        if self.visit_count == 0:
            return float('inf')
        
        # Q(s,a) + U(s,a)
        ucb = self.value() + c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return ucb

class MCTS:
    def __init__(self, net, config,device):
        self.net = net
        self.config = config
        self.device = device

    def reset(self):
        # MCTS类本身不需要重置状态，因为每次run都会创建新的搜索树
        pass

    def run(self, env, obs, add_noise=True):
        root = MCTSNode(parent=None, prior=1.0, action=None)
        
        # 获取神经网络预测
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(obs_tensor)
            policy = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        # Mask和归一化策略
        policy_masked = np.zeros_like(policy)
        if len(valid_actions) > 0:
            # 只保留有效动作的概率
            valid_probs = policy[valid_actions]
            if valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
            else:
                valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
            
            policy_masked[valid_actions] = valid_probs
            
            # 添加Dirichlet噪声
            if add_noise:
                noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(valid_actions))
                noise_weight = self.config['noise_weight']
                
                for i, a in enumerate(valid_actions):
                    policy_masked[a] = (1 - noise_weight) * policy_masked[a] + noise_weight * noise[i]
        
        # 创建子节点
        for a in valid_actions:
            root.children[a] = MCTSNode(parent=root, prior=policy_masked[a], action=a)
        root.is_expanded = True
        
        # 运行模拟
        for _ in range(self.config['num_mcts_simulations']):
            env_copy = env.clone()
            self._simulate(env_copy, root)
        
        # 收集访问次数
        visits = np.zeros(self.config['board_size'] ** 2)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        return visits

    def _simulate(self, env, node):
        if not node.is_expanded:
            obs = env._get_observation()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, value = self.net(obs_tensor)
                policy = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            if env.is_game_over():
                reward = env.get_reward()
                return -reward

            valid_actions = env.get_valid_actions()
            for a in valid_actions:
                node.children[a] = MCTSNode(parent=node, prior=policy[a], action=a)
            node.is_expanded = True
            
            return -value.item()
        
        best_child = None
        best_score = -float('inf')
        for child in node.children.values():
            score = child.ucb_score(self.config['c_puct'])
            if score > best_score:
                best_score = score
                best_child = child
        
        obs, reward, terminated, truncated, info = env.step(best_child.action)
        
        if terminated:
            value_for_child = reward 
        else:
            value_for_child = self._simulate(env, best_child)

        best_child.visit_count += 1
        best_child.value_sum += value_for_child 

        return -value_for_child