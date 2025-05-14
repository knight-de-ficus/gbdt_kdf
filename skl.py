import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import draw
import feature_select

def main():
    train_path = './data_3/kddtest_f.csv'
    test_path = './data_3/kddtrain_f.csv'

    # train_path = './data_3/kddtest_onehot.csv'
    # test_path = './data_3/kddtrain_onehot.csv'


    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df, test_df = feature_select.select_features(train_df, test_df, keep_ratio=0.5, save_path="selected_features_gbdt.png")

    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)

    

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(y_test))))
    print(cm)
    draw.draw_confusion_matrix(y_test, y_pred, label_name=[str(l) for l in sorted(set(y_test))], title="sklearn GBDT Confusion Matrix", pdf_save_path="sklearn_gbdt_confusion_matrix.png", dpi=300)

    # 随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print(classification_report(y_test, rf_pred))
    rf_cm = confusion_matrix(y_test, rf_pred, labels=sorted(list(set(y_test))))
    print(rf_cm)
    draw.draw_confusion_matrix(y_test, rf_pred, label_name=[str(l) for l in sorted(set(y_test))], title="sklearn RF Confusion Matrix", pdf_save_path="sklearn_rf_confusion_matrix.png", dpi=300)

if __name__ == "__main__":
    main()
