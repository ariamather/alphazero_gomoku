import torch
import numpy as np
import argparse
import random
import time
import os

# 假设 network.py, mcts.py, GomokuGame.py 在同一个目录下
from GomokuGame import GomokuEnv, Player
from network import AlphaZeroNet
from mcts import MCTS

class AlphaZeroAgent:
    """使用AlphaZero模型和MCTS进行决策的AI代理"""
    def __init__(self, model_path, name="AI"):
        """
        初始化代理，加载模型和配置。
        Args:
            model_path (str): 训练好的模型检查点路径 (.pt 文件)
            name (str): 代理的名称，用于显示
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.name} 正在使用设备: {self.device}")

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # 初始化网络并加载权重
        self.net = AlphaZeroNet(
            board_size=self.config['board_size'],
            num_res_blocks=self.config['num_res_blocks'],
            num_channels=self.config['num_channels']
        ).to(self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()  # 设置为评估模式
        print(f"{self.name} 模型加载成功！")

    def select_action(self, env, obs):
        """
        根据当前观察到的状态选择最佳动作。
        Args:
            env (GomokuEnv): 当前的游戏环境实例。
            obs (np.array): 当前状态的观察值。
        Returns:
            int: 选出的最佳动作。
        """
        device = torch.device(self.config['device'])
        # 创建MCTS实例
        mcts = MCTS(self.net, self.config , device)
        
        # 运行MCTS搜索
        # 在测试时不添加噪声 (add_noise=False) 以获得最强决策
        visits = mcts.run(env.clone(), obs, add_noise=False)
        
        # 选择访问次数最多的动作
        action = np.argmax(visits)
        return action

def run_self_play(agent):
    """
    运行AI自我对弈。
    Args:
        agent (AlphaZeroAgent): 配置好的AI代理。
    """
    board_size = agent.config['board_size']
    env = GomokuEnv(board_size=board_size, render_mode="human")
    obs, info = env.reset()

    print("\n======== AI 自我对弈开始 ========")
    
    while True:
        player_name = "黑棋 (AI)" if env.current_player == Player.BLACK else "白棋 (AI)"
        print(f"\n轮到 {player_name}...")
        
        action = agent.select_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 暂停一下，方便观察
        time.sleep(1)

        if terminated or truncated:
            env.render() # 渲染最终棋盘
            print("\n======== 游戏结束 ========")
            if env.winner == Player.NONE:
                print("结果: 平局！")
            else:
                winner_name = "黑棋 (AI)" if env.winner == Player.BLACK else "白棋 (AI)"
                print(f"结果: {winner_name} 获胜！")
            break

def run_model_vs_model(agent1, agent2, num_games=1, display_board=True, delay=1.0):
    """
    运行两个不同模型之间的对弈。
    Args:
        agent1 (AlphaZeroAgent): 第一个AI代理（执黑棋）。
        agent2 (AlphaZeroAgent): 第二个AI代理（执白棋）。
        num_games (int): 对弈的局数。
        display_board (bool): 是否显示棋盘。
        delay (float): 每步之间的延迟时间（秒）。
    Returns:
        dict: 对战结果统计
    """
    # 确保两个模型的棋盘大小一致
    if agent1.config['board_size'] != agent2.config['board_size']:
        raise ValueError("两个模型的棋盘大小必须一致！")
    
    board_size = agent1.config['board_size']
    
    # 统计结果
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'game_details': []
    }
    
    print(f"\n======== {agent1.name} vs {agent2.name} 对弈开始 ========")
    print(f"将进行 {num_games} 局对弈")
    print(f"{agent1.name} 执黑棋，{agent2.name} 执白棋\n")
    
    for game_num in range(num_games):
        print(f"\n--- 第 {game_num + 1} 局 ---")
        
        render_mode = "human" if display_board else None
        env = GomokuEnv(board_size=board_size, render_mode=render_mode)
        obs, info = env.reset()
        
        move_count = 0
        game_moves = []
        
        while True:
            move_count += 1
            
            # 根据当前玩家选择对应的代理
            if env.current_player == Player.BLACK:
                current_agent = agent1
                player_name = f"黑棋 ({agent1.name})"
            else:
                current_agent = agent2
                player_name = f"白棋 ({agent2.name})"
            
            print(f"\n第 {move_count} 手 - {player_name} 思考中...")
            
            # 选择动作
            action = current_agent.select_action(env, obs)
            x, y = env.action_to_human(action)
            print(f"{player_name} 落子于: {x} {y}")
            game_moves.append((env.current_player, x, y))
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            if display_board:
                env.render()
                time.sleep(delay)
            
            if terminated or truncated:
                print(f"\n第 {game_num + 1} 局结束！")
                
                # 记录结果
                if env.winner == Player.NONE:
                    print("结果: 平局！")
                    results['draws'] += 1
                    winner = "draw"
                elif env.winner == Player.BLACK:
                    print(f"结果: {agent1.name} (黑棋) 获胜！")
                    results['agent1_wins'] += 1
                    winner = agent1.name
                else:
                    print(f"结果: {agent2.name} (白棋) 获胜！")
                    results['agent2_wins'] += 1
                    winner = agent2.name
                
                results['game_details'].append({
                    'game_num': game_num + 1,
                    'winner': winner,
                    'total_moves': move_count,
                    'moves': game_moves
                })
                
                break
    
    # 显示总结
    print("\n======== 对弈结果总结 ========")
    print(f"总局数: {num_games}")
    print(f"{agent1.name} 胜利: {results['agent1_wins']} 局 ({results['agent1_wins']/num_games*100:.1f}%)")
    print(f"{agent2.name} 胜利: {results['agent2_wins']} 局 ({results['agent2_wins']/num_games*100:.1f}%)")
    print(f"平局: {results['draws']} 局 ({results['draws']/num_games*100:.1f}%)")
    
    return results

def run_human_vs_ai(agent, human_first=True):
    """
    运行人机对战。
    Args:
        agent (AlphaZeroAgent): 配置好的AI代理。
        human_first (bool): 人类是否先手（执黑棋）。
    """
    board_size = agent.config['board_size']
    env = GomokuEnv(board_size=board_size, render_mode="human")
    obs, info = env.reset()
    
    human_player = Player.BLACK if human_first else Player.WHITE
    ai_player = Player.WHITE if human_first else Player.BLACK
    
    human_name = "(黑棋)" if human_first else "(白棋)"
    ai_name = "(白棋)" if human_first else "(黑棋)"

    print("\n======== 人机对战开始 ========")
    print(f"您执 {human_name}, AI 执 {ai_name}")
    print("在提示符后输入坐标，例如 '7 7'")

    while True:
        if env.current_player == human_player:
            # 人类玩家回合
            env.render()
            valid_actions = env.get_valid_actions()
            while True:
                try:
                    move_input = input(f"请输入您的落子坐标 (行 列): ").strip()
                    if move_input.lower() in ['quit', 'exit']:
                        print("游戏已退出。")
                        return

                    x, y = map(int, move_input.split())
                    action = env.human_to_action(x, y)
                    
                    if action in valid_actions:
                        break
                    else:
                        print("非法移动或位置已被占据，请重新输入！")

                except (ValueError, IndexError):
                    print("输入格式错误，请输入两个由空格隔开的数字 (例如: 7 7)。")
        else:
            # AI 玩家回合
            print("\nAI 正在思考...")
            action = agent.select_action(env, obs)
            x, y = env.action_to_human(action)
            print(f"AI 落子于: {x} {y}")

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            env.render() # 渲染最终棋盘
            print("\n======== 游戏结束 ========")
            if env.winner == Player.NONE:
                print("结果: 平局！")
            elif env.winner == human_player:
                print("恭喜您，您获胜了！")
            else:
                print("很遗憾，AI 获胜了。")
            break

def main():
    parser = argparse.ArgumentParser(description="测试已训练的五子棋 AlphaZero 模型")
    parser.add_argument("--checkpoint", type=str, help="模型检查点文件的路径 (.pt)")
    parser.add_argument("--checkpoint2", type=str, help="第二个模型检查点文件的路径 (.pt)，用于模型对战")
    parser.add_argument("--mode", type=str, choices=["self-play", "human-vs-ai", "model-vs-model"], 
                        default="human-vs-ai", help="选择测试模式")
    parser.add_argument("--human-first", action="store_true", help="在人机对战模式中，人类是否先手 (默认AI先手)")
    parser.add_argument("--num-games", type=int, default=1, help="模型对战的局数 (默认1局)")
    parser.add_argument("--no-display", action="store_true", help="在模型对战中不显示棋盘")
    parser.add_argument("--delay", type=float, default=1.0, help="模型对战中每步之间的延迟时间（秒）")
    parser.add_argument("--name1", type=str, default="Model1", help="第一个模型的名称")
    parser.add_argument("--name2", type=str, default="Model2", help="第二个模型的名称")

    args = parser.parse_args()

    # 根据模式运行
    if args.mode == 'model-vs-model':
        if not args.checkpoint or not args.checkpoint2:
            parser.error("模型对战模式需要提供两个模型文件路径 (--checkpoint 和 --checkpoint2)")
        
        # 初始化两个AI代理
        agent1 = AlphaZeroAgent(model_path=args.checkpoint, name=args.name1)
        agent2 = AlphaZeroAgent(model_path=args.checkpoint2, name=args.name2)
        
        # 运行模型对战
        run_model_vs_model(
            agent1, 
            agent2, 
            num_games=args.num_games,
            display_board=not args.no_display,
            delay=args.delay
        )
    
    else:
        if not args.checkpoint:
            parser.error("需要提供模型文件路径 (--checkpoint)")
        
        # 初始化AI代理
        agent = AlphaZeroAgent(model_path=args.checkpoint)
        
        if args.mode == 'self-play':
            run_self_play(agent)
        elif args.mode == 'human-vs-ai':
            run_human_vs_ai(agent, human_first=args.human_first)

if __name__ == "__main__":
    main()