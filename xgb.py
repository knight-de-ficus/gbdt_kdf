import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import draw
import feature_select

def main():
    # train_path = './data_3/kddtest_f.csv'
    # test_path = './data_3/kddtrain_f.csv'

    # train_path = './data_3/kddtest_onehot.csv'
    # test_path = './data_3/kddtrain_onehot.csv'

    train_path = "./data_3/unswtrain_2f.csv"
    test_path = "./data_3/unswtest_2f.csv"
    
    train_raw_df = pd.read_csv(train_path)
    test_raw_df = pd.read_csv(test_path)

    print('train_raw_df各列的数据类型:')
    print(train_raw_df.dtypes)

    # 特征选择，保留80%最优特征
    train_df, test_df = feature_select.select_features(train_raw_df, test_raw_df, keep_ratio=0.5, save_path="selected_features_xgb.png")
    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)


    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(y_test))))
    print(cm)
    draw.draw_confusion_matrix(y_test, y_pred, label_name=[str(l) for l in sorted(set(y_test))], title="XGBoost Confusion Matrix", pdf_save_path="xgb_confusion_matrix.png", dpi=300)

if __name__ == "__main__":
    main()
