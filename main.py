import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data
from gbdt.data import DataSet
from gbdt.model import GBDT
from gbdt.gbdt_model import GBDTMultiClassifier
import feature_select as fs

def main():
    
    training_file_path = "./data_3/kddtrain_f.csv"
    test_file_path = "./data_3/kddtest_f.csv"

    # training_file_path = "./data_3/kddtrain_onehot.csv"
    # test_file_path = "./data_3/kddtest_onehot.csv"
    
    # training_file_path = "./data_3/unswtrain_1f.csv"
    # test_file_path = "./data_3/unswtest_1f.csv"


    
    training_data = pd.read_csv(test_file_path)
    test_data = pd.read_csv(training_file_path)

    training_data, test_data = fs.select_features(training_data, test_data,keep_ratio=0.8)

    training_labels = training_data.iloc[:, -1]
    training_features = training_data.iloc[:, 1:-1]
    test_labels = test_data.iloc[:, -1]
    test_features = test_data.iloc[:, 1:-1]

    # label_mapping = {'dos': 0, 'normal': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}

    # 使用自定义的GBDT多分类模型
    # gbdt_model = GBDTMultiClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, task='binary')
    gbdt_model = GBDTMultiClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,task='multiclass')

    print("GBDTMultiClassifier initialized.")

    # 训练模型
    fit_time_df = gbdt_model.fit(training_features.values, training_labels.values)
    print("GBDTMultiClassifier training completed.")

    # 保存每棵树训练时间到文件
    fit_time_df.to_csv("./gbdt_train_time.csv", index=False)
    print("Tree training time saved to './gbdt_train_time.csv'")

    # 保存模型到文件
    # joblib.dump(gbdt_model, "./gbdt_model_500.pkl")
    # print("Trained GBDTMultiClassifier model saved.")

    # 测试模型
    test_predictions = gbdt_model.predict(test_features.values)

    # 计算准确率
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"GBDTMultiClassifier Model Accuracy: {accuracy:.4f}")

    # 保存结果到CSV文件
    result_df = pd.DataFrame({
        "True Label": test_labels,
        "Predicted Label": test_predictions,
        "Match": (test_labels == test_predictions).astype(int)
    })
    result_df.to_csv("./result.csv", index=False)
    print("Results saved to './result.csv'")


if __name__ == "__main__":
    main()