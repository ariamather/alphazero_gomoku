import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


class Player(Enum):
    NONE = 0
    BLACK = 1
    WHITE = -1


class GomokuEnv(gym.Env):
    """五子棋环境，使用Gymnasium接口"""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, board_size=15, render_mode=None):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.render_mode = render_mode

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, board_size, board_size),
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
        """重置游戏环境（Gymnasium新接口）"""
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
        """执行一步动作（返回5个值以符合Gymnasium标准）"""
        if self.done:
            raise ValueError("游戏已结束，请调用reset()重新开始")

        # 将一维动作转换为二维坐标
        x, y = action // self.board_size, action % self.board_size

        # 检查动作是否合法
        if not self._is_valid_action(x, y):
            # 非法动作给予惩罚
            observation = self._get_observation()
            reward = -5.0
            terminated = False
            truncated = False
            info = {"invalid_move": True, "winner": None}
            return observation, reward, terminated, truncated, info

        # 执行动作
        self.board[x, y] = self.current_player.value
        self.last_action = (x, y)
        self.move_count += 1

        # 检查游戏结果
        if self._check_win(x, y):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
            terminated = True
            truncated = False
        elif self.move_count >= self.board_size * self.board_size:
            self.done = True
            self.winner = Player.NONE
            reward = 0.0
            terminated = True
            truncated = False
        else:
            reward = 0.0
            terminated = False
            truncated = False
            # 切换玩家
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """获取当前状态的观察值"""
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)

        # 第0层：当前玩家的棋子
        obs[0] = (self.board == self.current_player.value).astype(np.float32)

        # 第1层：对手的棋子
        opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        obs[1] = (self.board == opponent.value).astype(np.float32)

        # 第2层：最后一步的位置
        if self.last_action is not None:
            obs[2, self.last_action[0], self.last_action[1]] = 1

        return obs

    def is_game_over(self):
        return self.done

    def _get_info(self):
        """获取额外信息"""
        return {
            "current_player": self.current_player,
            "move_count": self.move_count,
            "winner": self.winner,
            "last_action": self.last_action,
            "valid_actions": self.get_valid_actions()
        }

    def _is_valid_action(self, x, y):
        """检查动作是否合法"""
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0

    def _check_win(self, x, y):
        """检查是否获胜"""
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            # 正向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                nx += dx
                ny += dy

            # 反向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                count += 1
                nx -= dx
                ny -= dy

            if count >= 5:
                return True

        return False

    def get_valid_actions(self):
        """获取所有合法动作"""
        return np.where(self.board.flatten() == 0)[0]

    def render(self):
        """渲染棋盘"""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()

    def clone(self):
        """
        高效浅拷贝环境状态，用于MCTS模拟。
        只拷贝必要属性（如棋盘、当前玩家），避免deepcopy的开销。
        """
        # 创建新实例，但不设置render_mode，避免在模拟时渲染
        new_env = GomokuEnv(self.board_size, render_mode=None)  # ← 修改这里！
        
        # 初始化新环境
        new_env.reset()
        
        # 拷贝棋盘（使用np.copy创建独立数组）
        new_env.board = np.copy(self.board)
        
        # 拷贝其他游戏状态（简单赋值）
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.last_action = self.last_action
        new_env.move_count = self.move_count
        
        return new_env

    def _render_human(self):
        """人类可读的渲染"""
        symbols = {0: '·', 1: '●', -1: '○'}

        # 清屏（可选）
        print("\033[2J\033[H")

        # 打印列标
        print('   ', end='')
        for i in range(self.board_size):
            print(f'{i:2}', end=' ')
        print()

        # 打印棋盘
        for i in range(self.board_size):
            print(f'{i:2} ', end='')
            for j in range(self.board_size):
                symbol = symbols[self.board[i, j]]
                if self.last_action == (i, j):
                    print(f'[{symbol}]', end='')
                else:
                    print(f' {symbol} ', end='')
            print()

        print(f"\n当前玩家: {'黑棋' if self.current_player == Player.BLACK else '白棋'}")
        print(f"总步数: {self.move_count}")

        if self.done:
            if self.winner == Player.NONE:
                print("游戏结束：平局！")
            else:
                print(f"游戏结束：{'黑棋' if self.winner == Player.BLACK else '白棋'}获胜！")

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
        """清理资源"""
        pass

    def human_to_action(self, x, y):
        """将人类输入的坐标转换为动作"""
        return x * self.board_size + y

    def action_to_human(self, action):
        """将动作转换为人类可读的坐标"""
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
