import os
import sys
import torch
import numpy
import traceback

# 不需要添加安全全局变量，因为我们使用的是修复后的模型

import numpy as np
from flask import Flask, render_template, request, jsonify
import numpy._core.multiarray
# 将项目根目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from GomokuGame import GomokuEnv, Player
from network import AlphaZeroNet
from mcts import MCTS

app = Flask(__name__)

# --- 全局变量和模型加载 ---
BOARD_SIZE = 12
N_MCTS_SIMS = 400 # 为加快响应速度，可以减少模拟次数

# 创建MCTS配置
mcts_config = {
    'c_puct': 1.0,
    'num_mcts_simulations': N_MCTS_SIMS,
    'board_size': BOARD_SIZE,
    'dirichlet_alpha': 0.03,  # 默认值，可根据需要调整
    'noise_weight': 0.25      # 默认值，可根据需要调整
}

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用修复后的模型
model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please make sure 'best.pt' exists in the root directory.")

net = AlphaZeroNet(board_size=BOARD_SIZE, num_res_blocks=12, num_channels=256).to(device)
try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("Model loaded with weights_only=False (not recommended for production)")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading with weights_only=False: {e}")
    try:
        with torch.serialization.safe_globals([numpy._core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print("Model loaded successfully without weights_only")
    except Exception as e2:
        print(f"Error loading model: {e2}")
        raise

net.eval()

mcts = MCTS(net, mcts_config, device)

# 创建一个游戏实例
env = GomokuEnv(board_size=BOARD_SIZE)
# 初始化游戏环境
env.reset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def make_move():
    data = request.get_json()
    if not data or 'row' not in data or 'col' not in data:
        return jsonify({'error': 'Invalid request: missing row or col'}), 400

    player_row = data['row']
    player_col = data['col']
    
    print(f"Received move request: ({player_row}, {player_col})")
    print(f"Current player: {env.current_player}")
    print(f"Move count: {env.move_count}")
    
    # 检查游戏是否已经结束
    if env.is_game_over():
        print("Game is already over")
        return jsonify({
            'board': env.board.tolist(),
            'ai_action': -1,
            'game_over': True,
            'winner': env.winner.value if env.winner else 0
        })
    
    # 检查位置是否有效
    if player_row < 0 or player_row >= BOARD_SIZE or player_col < 0 or player_col >= BOARD_SIZE:
        return jsonify({'error': 'Invalid position'}), 400
    
    # 检查位置是否为空
    if env.board[player_row, player_col] != 0:
        return jsonify({'error': 'Position already occupied'}), 400
    
    # 执行玩家的移动
    player_action = player_row * BOARD_SIZE + player_col
    print(f"Player {env.current_player} moving to action {player_action}")
    env.step(player_action)
    
    # 检查游戏是否结束（玩家获胜）
    if env.is_game_over():
        game_over = True
        winner = env.winner.value if env.winner else 0
        print(f"Game over after player move. Winner: {winner}")
        response = {
            'board': env.board.tolist(),
            'ai_action': -1,
            'game_over': game_over,
            'winner': winner
        }
        return jsonify(response)
    
    # AI下棋
    print(f"AI {env.current_player} thinking...")
    
    # 获取当前观察状态
    observation = env._get_observation()
    
    # 运行MCTS获取访问次数
    visits = mcts.run(env, observation)
    
    # 选择访问次数最多的动作
    ai_action = np.argmax(visits)
    
    ai_row = ai_action // BOARD_SIZE
    ai_col = ai_action % BOARD_SIZE
    print(f"AI moving to ({ai_row}, {ai_col}), action {ai_action}")
    
    # 确保AI选择的位置是有效的
    if env.board[ai_row, ai_col] != 0:
        print(f"Warning: AI selected occupied position ({ai_row}, {ai_col})")
        # 找一个有效的位置
        valid_actions = np.where(env.board.flatten() == 0)[0]
        if len(valid_actions) > 0:
            ai_action = valid_actions[0]
            ai_row = ai_action // BOARD_SIZE
            ai_col = ai_action % BOARD_SIZE
            print(f"AI moving to alternative position ({ai_row}, {ai_col})")
        else:
            print("No valid moves available")
            return jsonify({
                'board': env.board.tolist(),
                'ai_action': -1,
                'game_over': True,
                'winner': 0  # Draw
            })
    
    # 执行AI的移动
    env.step(ai_action)
    
    # 检查游戏是否结束
    game_over = env.is_game_over()
    winner = env.winner.value if env.winner else 0
    
    if game_over:
        print(f"Game over after AI move. Winner: {winner}")

    response = {
        'board': env.board.tolist(),
        'ai_action': int(ai_action),
        'game_over': game_over,
        'winner': winner
    }
    return jsonify(response)

@app.route('/init', methods=['GET'])
def init_game():
    try:
        print("Initializing new game")  # 调试输出
        env.reset()
        mcts.reset()
        current_player = env.current_player.value
        board_list = env.board.tolist()
        print(f"Init successful: current_player={current_player}, board_size={len(board_list)}")  # 调试输出
        return jsonify({
            'board': board_list,
            'current_player': current_player
        })
    except Exception as e:
        print(f"Error in /init: {str(e)}")  # 服务器端日志
        return jsonify({'error': f'Failed to initialize: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset_game():
    try:
        print("Resetting game")
        env.reset()
        mcts.reset()
        return jsonify({'board': env.board.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 全局错误处理函数
@app.errorhandler(Exception)
def handle_exception(e):
    # 记录异常堆栈信息
    print(traceback.format_exc())
    # 返回JSON格式的错误响应
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)