import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_data
from gbdt.data import DataSet
from gbdt.model import GBDT
from gbdt.gbdt_model import GBDTMultiClassifier
import draw

def main():
    
    training_file_path = "./data_3/kddtrain_f.csv"
    test_file_path = "./data_3/kddtest_f.csv"
    
    training_data = pd.read_csv(training_file_path)
    test_data = pd.read_csv(test_file_path)

    training_labels = training_data.iloc[:, -1]
    training_features = training_data.iloc[:, 1:-1]
    test_labels = test_data.iloc[:, -1]
    test_features = test_data.iloc[:, 1:-1]
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_features, training_labels)
    knn_pred = knn.predict(test_features)
    print("KNN 分类报告:")
    print(classification_report(test_labels, knn_pred))
    # 计算TP, TN, FP, FN
    knn_cm = confusion_matrix(test_labels, knn_pred, labels=list(set(test_labels)))
    print("KNN 混淆矩阵:")
    print(knn_cm)
    print("KNN TP:", knn_cm.diagonal().sum())
    print("KNN FP:", knn_cm.sum(axis=0) - knn_cm.diagonal())
    print("KNN FN:", knn_cm.sum(axis=1) - knn_cm.diagonal())
    print("KNN TN:", knn_cm.sum() - (knn_cm.sum(axis=0) + knn_cm.sum(axis=1) - knn_cm.diagonal()))
    draw.draw_confusion_matrix(test_labels, knn_pred, label_name=[str(l) for l in sorted(set(test_labels))], title="KNN Confusion Matrix", pdf_save_path="knn_confusion_matrix.png", dpi=300)

    # 逻辑回归
    lr = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
    lr.fit(training_features, training_labels)
    lr_pred = lr.predict(test_features)
    print("逻辑回归 分类报告:")
    print(classification_report(test_labels, lr_pred))
    lr_cm = confusion_matrix(test_labels, lr_pred, labels=list(set(test_labels)))
    print("逻辑回归 混淆矩阵:")
    print(lr_cm)
    print("逻辑回归 TP:", lr_cm.diagonal().sum())
    print("逻辑回归 FP:", lr_cm.sum(axis=0) - lr_cm.diagonal())
    print("逻辑回归 FN:", lr_cm.sum(axis=1) - lr_cm.diagonal())
    print("逻辑回归 TN:", lr_cm.sum() - (lr_cm.sum(axis=0) + lr_cm.sum(axis=1) - lr_cm.diagonal()))
    draw.draw_confusion_matrix(test_labels, lr_pred, label_name=[str(l) for l in sorted(set(test_labels))], title="LR Confusion Matrix", pdf_save_path="lr_confusion_matrix.png", dpi=300)

    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(training_features, training_labels)
    rf_pred = rf.predict(test_features)
    print("随机森林 分类报告:")
    print(classification_report(test_labels, rf_pred))
    rf_cm = confusion_matrix(test_labels, rf_pred, labels=list(set(test_labels)))
    print("随机森林 混淆矩阵:")
    print(rf_cm)
    print("随机森林 TP:", rf_cm.diagonal().sum())
    print("随机森林 FP:", rf_cm.sum(axis=0) - rf_cm.diagonal())
    print("随机森林 FN:", rf_cm.sum(axis=1) - rf_cm.diagonal())
    print("随机森林 TN:", rf_cm.sum() - (rf_cm.sum(axis=0) + rf_cm.sum(axis=1) - rf_cm.diagonal()))
    draw.draw_confusion_matrix(test_labels, rf_pred, label_name=[str(l) for l in sorted(set(test_labels))], title="RF Confusion Matrix", pdf_save_path="rf_confusion_matrix.png", dpi=300)
    
if __name__ == "__main__":
    main()