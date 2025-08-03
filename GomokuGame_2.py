import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


class Player(Enum):
    NONE = 0
    BLACK = 1
    WHITE = -1


class GomokuEnv(gym.Env):
    """【修改后】的五子棋环境，增加了奖励塑造和对称性数据增强功能"""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, board_size=15, render_mode=None):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.render_mode = render_mode

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 【修改】观察空间增加一个维度，用于表示当前玩家颜色
        # 第0层：当前玩家棋子
        # 第1层：对手玩家棋子
        # 第2层：最后落子位置
        # 第3层：当前玩家颜色 (1 for black, 0 for white)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(4, board_size, board_size), # 形状从3改为4
            dtype=np.float32
        )

        # 初始化游戏状态
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        self.last_action = None
        self.move_count = 0

    def reset(self, seed=None, options=None):
        """重置游戏环境"""
        super().reset(seed=seed)

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = Player.BLACK
        self.done = False
        self.winner = None
        self.last_action = None
        self.move_count = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """【修改】执行一步动作，并加入奖励塑造逻辑"""
        if self.done:
            raise ValueError("游戏已结束，请调用reset()重新开始")

        x, y = action // self.board_size, action % self.board_size

        if not self._is_valid_action(x, y):
            # 非法动作给予大的负奖励
            observation = self._get_observation()
            # 【修改】调整非法移动的惩罚值
            reward = -10.0
            terminated = True # 非法移动直接终止本局游戏
            truncated = False
            info = {"invalid_move": True, "winner": None}
            self.done = True # 【新增】将游戏状态设为完成
            return observation, reward, terminated, truncated, info

        # 执行动作
        self.board[x, y] = self.current_player.value
        self.last_action = (x, y)
        self.move_count += 1
        
        # ---- 【新增】奖励塑造的核心逻辑 ----
        reward = 0
        terminated = False
        truncated = False

        # 1. 检查当前玩家是否获胜
        if self._check_win(x, y):
            self.done = True
            self.winner = self.current_player
            reward = 15.0  # 获胜给予高奖励
            terminated = True
        
        # 2. 如果没有获胜，计算启发式奖励
        else:
            # 计算当前落子带来的进攻奖励
            offensive_reward = self._calculate_patterns_reward(x, y, self.current_player)
            
            # 计算当前落子带来的防守奖励（通过评估对手在此处落子能获得的奖励）
            opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            # 假设对手要下在这里，我们通过落子来“阻止”他获得奖励
            defensive_reward = self._calculate_patterns_reward(x, y, opponent)

            # 组合奖励：进攻奖励 + 防守奖励 + 每步的微小惩罚
            reward = offensive_reward + defensive_reward - 0.1 

        # 3. 检查是否平局
        if not terminated and self.move_count >= self.board_size * self.board_size:
            self.done = True
            self.winner = Player.NONE
            reward = 0.0  # 平局奖励为0
            terminated = True

        # 如果游戏尚未结束，切换玩家
        if not self.done:
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        
        # ------------------------------------

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        # 【修改】返回重构后的奖励和状态
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """【修改】获取当前状态的观察值，增加玩家颜色层"""
        obs = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)

        # 第0层：当前玩家的棋子
        obs[0] = (self.board == self.current_player.value).astype(np.float32)

        # 第1层：对手的棋子
        opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        obs[1] = (self.board == opponent.value).astype(np.float32)

        # 第2层：最后一步的位置
        if self.last_action is not None:
            obs[2, self.last_action[0], self.last_action[1]] = 1
            
        # 【新增】第3层：指示当前玩家颜色 (1=黑, 0=白)
        if self.current_player == Player.BLACK:
            obs[3] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        else:
            obs[3] = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        return obs

    # --- 【新增】用于奖励塑造的辅助函数 ---
    def _calculate_patterns_reward(self, x, y, player):
        """
        计算在(x, y)点落子后，对指定玩家形成的棋型（连二、活三、活四等）的奖励。
        这是奖励塑造的关键，用于评估一步棋的即时价值。
        """
        reward = 0
        patterns = {
            "live_four": 4.0,
            "dead_four": 2.0,
            "live_three": 1.0,
            "dead_three": 0.5,
            "live_two": 0.2,
            "dead_two": 0.1,
        }
        
        # 临时在棋盘上放置棋子以进行评估
        self.board[x, y] = player.value
        
        for p_name, p_reward in patterns.items():
            count = self._count_specific_pattern(x, y, player, p_name)
            reward += count * p_reward
            
        # 评估后恢复棋盘原状
        self.board[x, y] = 0
        
        return reward

    def _count_specific_pattern(self, x, y, player, pattern_name):
        """
        在(x, y)周围计算特定棋型（如“活三”）的数量。
        这是一个简化的版本，实际应用中可以设计得更精确。
        """
        count = 0
        player_val = player.value
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # 水平, 垂直, 右斜, 左斜

        for dx, dy in directions:
            # 检查每条线
            line = []
            for i in range(-4, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    line.append(self.board[nx, ny])
                else:
                    line.append(99) # 用一个边界值表示棋盘外

            # 将棋子值转换为字符串，便于匹配
            s_line = "".join(map(str, line)).replace(str(player_val), "P").replace(str(-player_val), "O").replace("0", "_")
            
            # 简化的模式匹配
            if pattern_name == "live_four" and "_PPPP_" in s_line: count += 1
            if pattern_name == "dead_four" and ("OPPPP_" in s_line or "_PPPPO" in s_line): count += 1
            if pattern_name == "live_three" and "_PPP_" in s_line: count += 1
            if pattern_name == "dead_three" and ("OPPP_" in s_line or "_PPPO" in s_line): count += 1
            if pattern_name == "live_two" and "_PP_" in s_line: count += 1
            if pattern_name == "dead_two" and ("OPP_" in s_line or "_PPO" in s_line): count += 1
                
        return count
    # ---------------------------------------

    def _is_valid_action(self, x, y):
        """检查动作是否合法"""
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0

    def is_game_over(self):
        return self.done

    def _check_win(self, x, y):
        """检查是否获胜"""
        player_val = self.board[x, y]
        if player_val == 0: return False # 【新增】确保检查的是有棋子的位置
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            # 正向检查
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player_val:
                    count += 1
                else:
                    break
            # 反向检查
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player_val:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        return False

    def get_valid_actions(self):
        """获取所有合法动作"""
        return np.where(self.board.flatten() == 0)[0]
        
    def clone(self):
        """
        高效浅拷贝环境状态，用于MCTS模拟。
        只拷贝必要属性（如棋盘、当前玩家），避免deepcopy的开销。
        """
        # 创建新实例，但不设置render_mode，避免在模拟时渲染
        new_env = GomokuEnv(self.board_size, render_mode=None)  # ← 修改这里！
        
        # 拷贝棋盘（使用np.copy创建独立数组）
        new_env.board = np.copy(self.board)
        
        # 拷贝其他游戏状态（简单赋值）
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.last_action = self.last_action
        new_env.move_count = self.move_count
        
        return new_env


    # --- 【新增】对称性数据增强函数 ---
    @staticmethod
    def get_symmetric_data(board, pi):
        """
        输入一个棋盘状态和对应的策略(pi, 一个与棋盘大小相同的概率分布向量),
        返回8个对称的 (board, pi) 数据对。
        这应该在你的训练循环中使用，而不是在环境内部。
        """
        board_size = board.shape[1]
        pi_board = pi.reshape(board_size, board_size)
        
        symmetric_data = []
        
        current_board = np.copy(board)
        current_pi_board = np.copy(pi_board)

        # 8种对称性：4次旋转 + 4次旋转后的水平翻转
        for i in range(4):
            # 旋转
            rotated_board = np.rot90(current_board, i, axes=(1, 2))
            rotated_pi = np.rot90(current_pi_board, i)
            symmetric_data.append((rotated_board, rotated_pi.flatten()))
            
            # 翻转
            flipped_board = np.flip(rotated_board, axis=2)
            flipped_pi = np.flip(rotated_pi, axis=1)
            symmetric_data.append((flipped_board, flipped_pi.flatten()))
            
        return symmetric_data

    # --- 以下是未作重大修改的渲染和其他辅助函数 ---
    
    def _get_info(self):
        """获取额外信息"""
        return {
            "current_player": self.current_player,
            "move_count": self.move_count,
            "winner": self.winner,
            "last_action": self.last_action,
            "valid_actions": self.get_valid_actions()
        }
        
    def render(self):
        """渲染棋盘"""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()

    def _render_human(self):
        """人类可读的渲染"""
        symbols = {0: '·', 1: '●', -1: '○'}
        print("\033[2J\033[H", end="") # 清屏
        print('   ', end='')
        for i in range(self.board_size):
            print(f'{i:<2}', end='')
        print()
        for i in range(self.board_size):
            print(f'{i:2} ', end='')
            for j in range(self.board_size):
                symbol = symbols[self.board[i, j]]
                if self.last_action == (i, j):
                    print(f'[{symbol}]', end='')
                else:
                    print(f' {symbol} ', end='')
            print()
        print(f"\n当前玩家: {'黑棋 (●)' if self.current_player == Player.BLACK else '白棋 (○)'}")
        print(f"总步数: {self.move_count}")
        if self.done:
            if self.winner == Player.NONE:
                print("游戏结束：平局！")
            else:
                winner_name = '黑棋 (●)' if self.winner == Player.BLACK else '白棋 (○)'
                print(f"游戏结束：{winner_name} 获胜！")

    def _render_ansi(self):
        """返回ANSI字符串"""
        symbols = {0: '·', 1: '●', -1: '○'}
        board_str = ""
        for i in range(self.board_size):
            for j in range(self.board_size):
                board_str += symbols[self.board[i, j]] + " "
            board_str += "\n"
        return board_str

    def close(self):
        pass

    def human_to_action(self, x, y):
        return x * self.board_size + y

    def action_to_human(self, action):
        return action // self.board_size, action % self.board_size


# 注册环境（可选）
gym.register(
    id='Gomoku-v0',
    entry_point='__main__:GomokuEnv',
    kwargs={'board_size': 15}
)


# 游戏控制器保持不变，但需要适配新的step返回值
class GomokuGame:
    """五子棋游戏控制器"""

    def __init__(self, board_size=15):
        self.env = GomokuEnv(board_size, render_mode="human")
        self.board_size = board_size

    def play_human_vs_ai(self, ai_agent, human_first=True):
        """人机对战"""
        obs, info = self.env.reset()

        human_player = Player.BLACK if human_first else Player.WHITE
        ai_player = Player.WHITE if human_first else Player.BLACK

        done = False
        while not done:
            if self.env.current_player == human_player:
                # 人类玩家回合
                while True:
                    try:
                        move_input = input("请输入坐标 (行 列): ").strip()
                        if move_input.lower() == 'quit':
                            print("游戏退出")
                            return None

                        x, y = map(int, move_input.split())
                        action = self.env.human_to_action(x, y)

                        obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated

                        if "invalid_move" not in info:
                            break
                        else:
                            print("非法移动，请重新输入！")
                    except (ValueError, IndexError):
                        print("输入格式错误，请输入两个数字，用空格分隔")
            else:
                # AI玩家回合
                print("AI正在思考...")
                valid_actions = info.get("valid_actions", self.env.get_valid_actions())
                action = ai_agent.select_action(obs, valid_actions)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                x, y = self.env.action_to_human(action)
                print(f"AI落子: {x} {y}")

        return self.env.winner


# 示例AI代理
class RandomAgent:
    """随机AI代理"""

    def select_action(self, obs, valid_actions):
        return np.random.choice(valid_actions)


# 使用示例
if __name__ == "__main__":
    # 测试环境
    env = GomokuEnv(board_size=9, render_mode="human")

    # 测试基本功能
    obs, info = env.reset()
    print("初始观察空间形状:", obs.shape)
    print("动作空间大小:", env.action_space.n)

    # 创建游戏并开始人机对战
    game = GomokuGame(board_size=9)
    random_ai = RandomAgent()

    print("五子棋游戏 - 人机对战")
    print("输入 'quit' 退出游戏")
    winner = game.play_human_vs_ai(random_ai, human_first=True)
