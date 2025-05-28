import pandas as pd
from sklearn.metrics import classification_report
import draw

import gbdt.LightGBM as lgb

import gbdt.feature_select as fs
from gbdt.gbdt_model import GBDTMultiClassifier
import time

def main():

#预处理文件后在这里调用
    train_path = './data_3/kddtrain_huffman.csv'
    test_path = './data_3/kddtest_huffman.csv'

    # train_path = './data_3/kddtrain_onehot.csv'
    # test_path = './data_3/kddtest_onehot.csv'

    # train_path = './data_3/unswtrain_2f.csv'
    # test_path = './data_3/unswtest_2f.csv'

    # train_path = './unswtrain_onehot_label.csv'
    # test_path = './unswtest_onehot_label.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

# 重采样
    train_df = fs.oversample_data(train_df)
    print("Oversampling done.")
    print("train_df shape:", train_df.shape)
    print("分布:")
    print(train_df.iloc[:, -1].value_counts())
    
# 卡方检验
    train_df, test_df = fs.chi2_select_features(train_df, test_df, keep_ratio=0.8, save_path="selected_features_gbdt.png")

    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

#前向选择
    # start_time = time.time()
    
    # max_features = int(X_train.shape[1] * 0.5)
    # best_features = fs.forward_feature_selection(X_train, y_train, X_test, y_test, max_features=max_features, step=5)
    # print('best features:', best_features)

    # end_time = time.time()
    # print(f"feature-select time: {end_time - start_time:.2f} ")

#nsl-kdd
    best_features = ['28', '33', '32', '27', '38', '103', '4', '102', '31', '20', '29', '21', '15', '22', '30', '11', '14', '201', '200', '9']

#unsw-nb15
    # best_features = ['3', '23', '4', '8', '24', '1', '20', '2', '52', '14', '6', '41', '29', '44', '31', '30', '46', '47', '39', '16', '53', '25', '35', '26', '17', '42', '43', '45', '21', '28', '40', '15', '48', '36', '12', '64', '7', '11', '13', '18']

    class_w = {
        '1': 0.1,
        '0': 0.6,
        '2': 0.6,
        '3': 2,
        '4': 1.2
    }
    
    # # class_w = {
    # #     '1': 1,
    # #     '0': 1,
    # #     '2': 1,
    # #     '3': 1,
    # #     '4': 1
    # # }

    # # class_w = {
    # #     '0': 0.1,
    # #     '1': 1.2,
    # #     '2': 1.2,
    # #     '3': 0.8,
    # #     '4': 1.2,
    # #     '5': 1,
    # #     '6': 0.8,
    # #     '7': 1,
    # #     '8': 1.2,
    # #     '9': 0.8
    # # }

#自定义的GBDT
    gbdt_model = GBDTMultiClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,task='multiclass')
    sample_w = fs.compute_sample_weight(class_weight=class_w, y=y_train)
    gbdt_model.fit(X_train[best_features].values, y_train.values, sample_weight=sample_w)
    test_predictions = gbdt_model.predict(X_test[best_features].values)

    print(classification_report(y_test, test_predictions))

# LightGBM
    # gbdt_model = lgb.GBDTMultiClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, task='multiclass')
    # gbdt_model.train(X_train[best_features].values, y_train.values, class_weight=class_w)
    # test_predictions = gbdt_model.predict(X_test[best_features].values)

    # print(classification_report(y_test, test_predictions))

    
    result_df = pd.DataFrame({
        "True Label": y_test,
        "Predicted Label": test_predictions,
        "Match": (y_test == test_predictions).astype(int)
    })
    result_df.to_csv("./result.csv", index=False)

if __name__ == "__main__":
    main()