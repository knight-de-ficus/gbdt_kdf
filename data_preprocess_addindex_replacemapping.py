import pandas as pd

def add_index_and_replace_labels(file_path, save_path, attack_types_path):
    # 读取数据
    df = pd.read_csv(file_path, header=None)

    # 添加列索引
    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # 读取攻击类型映射
    with open(attack_types_path, 'r') as f:
        attack_mapping = dict(line.strip().split() for line in f)

    # 替换最后一列的标签
    df[df.columns[-2]] = df[df.columns[-2]].replace(attack_mapping)

    # 保存修改后的文件
    df.to_csv(save_path, index=False, header=True)

if __name__ == "__main__":
    train_file_path = "./data_2/KDDTrain+.csv"
    test_file_path = "./data_2/KDDTest+.csv"
    save_path_train = "./data_2/kddtrain.csv"
    save_path_test = "./data_2/kddtest.csv"
    attack_types_path = "data_2/attack_types.txt"

    # 处理训练集和测试集
    add_index_and_replace_labels(train_file_path,save_path_test, attack_types_path)
    add_index_and_replace_labels(test_file_path,save_path_train, attack_types_path)



