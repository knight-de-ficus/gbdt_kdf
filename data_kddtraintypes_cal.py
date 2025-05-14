import pandas as pd


if __name__ == "__main__":
    # train_file_path = "./data_2/KDDTrain+.csv"
    train_file_path = "./data_3/kddtrain_f.csv"

    # 读取数据
    df = pd.read_csv(train_file_path)

    # 构造字典，将第二列的所有字符串映射到整数
    unique_values = df.iloc[:, -1].unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}

    # 标准输出字典
    print(value_to_int)

