import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_from_json(file_path):
    """
    从 JSON 文件中读取 "loss" 列表并绘制一张折线图。

    参数:
    file_path (str): JSON 文件的路径。
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解析 {file_path} 中的 JSON 数据")
        return

    if "loss" not in data or not isinstance(data["loss"], list):
        print("错误：JSON 文件中缺少 'loss' 键，或者其值不是一个列表。")
        return

    data = data["policy_loss"]
    
    # 确保列表中的所有数据都是数值类型 (整数或浮点数)
    if not all(isinstance(n, (int, float)) for n in data):
        print("错误：'loss' 列表包含非数值类型的数据。")
        return
        
    # 使用 pandas 创建一个 DataFrame，方便绘图
    # 横坐标为列表的索引，纵坐标为列表中的值
    df = pd.DataFrame({
        'Index': range(len(data)),
        'Value': data
    })

    # 开始绘图
    plt.figure(figsize=(10, 6)) # 设置图形大小
    plt.plot(df['Index'], df['Value'], marker='o', linestyle='-', color='b')

    # 添加标题和坐标轴标签
    plt.title("loss Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True) # 显示网格

    # 保存图形
    output_filename = "iteration_plot.png"
    plt.savefig(output_filename)
    print(f"图形已保存为: {output_filename}")


# 2. 调用函数并传入您的 JSON 文件名
plot_loss_from_json("stats.json")