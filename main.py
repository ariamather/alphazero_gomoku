# main.py
import torch.multiprocessing as mp
from train2 import AlphaZeroTrainer
from config import CONFIG

def main():
    """主函数"""
    # 打印核心配置信息
    print("===== AlphaZero Gomoku Training =====")
    print(f"Board Size: {CONFIG['board_size']}x{CONFIG['board_size']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Number of MCTS Simulations: {CONFIG['num_mcts_simulations']}")
    print("=====================================")

    # 初始化并开始训练
    trainer = AlphaZeroTrainer(config=CONFIG)

    # 如果需要从检查点恢复训练，取消下面这行的注释
    # trainer.load_checkpoint('path/to/your/checkpoint.pt')

    trainer.run()

if __name__ == "__main__":
    # 在Windows或macOS上，使用多进程时建议使用 'spawn' 或 'forkserver'
    # 这行代码最好放在 `if __name__ == "__main__":` 里面，且在所有多进程代码之前
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    
    main()