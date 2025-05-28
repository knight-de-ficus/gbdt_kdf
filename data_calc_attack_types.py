import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def draw(training_file_path,label_dict):
    # training_file_path = "./data_3/kddtrain_huffman.csv"
    # label_dict = {0: 'dos', 1: 'normal', 2: 'probe', 3: 'r2l', 4: 'u2r'}

    # 读取训练集
    df = pd.read_csv(training_file_path)
    label_col = df.columns[-1]
    label_counts = df[label_col].value_counts()
    print("原始标签统计：")
    print(label_counts)

    # 标签转换
    df['label_mapped'] = df[label_col].map(label_dict)
    mapped_counts = df['label_mapped'].value_counts().sort_index()
    print("映射后标签统计：")
    print(mapped_counts)
    print("mapped_counts是否有数据：", not mapped_counts.empty)

    # 检查NaN
    nan_count = df['label_mapped'].isna().sum()
    print(f"映射后NaN数量: {nan_count}")
    if mapped_counts.empty:
        print("没有可用的标签数据，无法绘制柱状图！")
        return

    # 柱状图展示（对数计数）
    plt.figure(figsize=(8, 6))
    # x轴顺序严格按照label_dict顺序
    x_keys = list(label_dict.values())  # 直接取name顺序
    labels = x_keys
    # 保证mapped_counts顺序与x_keys一致
    mapped_counts_full = [mapped_counts.get(k, 0) for k in x_keys]
    log_counts = [np.log10(x) if x > 0 else 0 for x in mapped_counts_full]
    bars = plt.bar(labels, log_counts, color='skyblue')
    # 添加数据标签（原始计数）
    for bar, count in zip(bars, mapped_counts_full):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count}', ha='center', va='bottom', fontsize=16)
    plt.xlabel('流量类型', fontsize=20)
    plt.ylabel('log10(样本数)', fontsize=20)
    plt.title('流量类型log10样本数分布（映射后）', fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('bar_log_debug.png')
    plt.show()

if __name__ == "__main__":

    training_file_path = "./data_3/kddtrain_huffman.csv"
    label_dict = {0: 'dos', 1: 'normal', 2: 'probe', 3: 'r2l', 4: 'u2r'}

    draw(training_file_path,label_dict)

    # training_file_path = "./data_3/unswtrain_2f.csv"
    # label_dict = {0: 'Normal', 1: 'Backdoor', 2: 'Analysis', 3: 'Fuzzers', 4: 'Shellcode', 5: 'Reconnaissance', 6: 'Exploits', 7: 'DoS', 8: 'Worms', 9: 'Generic'}


    # draw(training_file_path,label_dict)
