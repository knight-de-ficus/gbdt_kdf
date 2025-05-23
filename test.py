import pandas as pd

def main():
    train_path = './data/UNSW_NB15_training-set.csv'
    test_path = './data/UNSW_NB15_testing-set.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    dict_unsw = {'Normal': 0, 'Backdoor': 1, 'Analysis': 2, 'Fuzzers': 3, 'Shellcode': 4, 'Reconnaissance': 5, 'Exploits': 6, 'DoS': 7, 'Worms': 8, 'Generic': 9}   # 只对object类型（类别型）做one-hot
    # 用dict_unsw将倒数第二列（标签）进行映射
    label_col = train_df.columns[-2]
    train_df[label_col] = train_df[label_col].map(dict_unsw)
    test_df[label_col] = test_df[label_col].map(dict_unsw)

    # 特征one-hot编码（不包含标签和索引列）
    feature_cols = train_df.columns[1:-2]  # 排除第一列（如索引）、最后两列（标签和原始标签）
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all_onehot = pd.get_dummies(X_all)
    X_train_onehot = X_all_onehot.iloc[:len(X_train), :].reset_index(drop=True)
    X_test_onehot = X_all_onehot.iloc[len(X_train):, :].reset_index(drop=True)

    # 取映射后的标签
    y_train = train_df[label_col].reset_index(drop=True)
    y_test = test_df[label_col].reset_index(drop=True)

    # 合并输出
    train_out = pd.concat([X_train_onehot, y_train], axis=1)
    test_out = pd.concat([X_test_onehot, y_test], axis=1)
    train_out.to_csv('unswtrain_onehot_label.csv', index=False)
    test_out.to_csv('unswtest_onehot_label.csv', index=False)
    print('已保存one-hot特征+映射标签的训练集和测试集')



if __name__ == "__main__":
    main()