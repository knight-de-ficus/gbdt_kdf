import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import draw
import feature_select as fs

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
# import lightgbm as lgb
import gbdtmodel_1 as lgb
import time

from gbdt.gbdt_model import GBDTMultiClassifier


def main():


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

    train_df = fs.oversample_data(train_df)
    print("Oversampling done.")
    print("train_df shape:", train_df.shape)
    print("标签分布:")
    print(train_df.iloc[:, -1].value_counts())
    
    train_df, test_df = fs.chi2_select_features(train_df, test_df, keep_ratio=0.8, save_path="selected_features_gbdt.png")

    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]



    # gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # gbdt.fit(X_train, y_train)
    # y_pred = gbdt.predict(X_test)

    

    # print(classification_report(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(y_test))))
    # print(cm)
    # draw.draw_confusion_matrix(y_test, y_pred, label_name=[str(l) for l in sorted(set(y_test))], title="sklearn GBDT Confusion Matrix", pdf_save_path="sklearn_gbdt_confusion_matrix.png", dpi=300)

    # # 随机森林模型
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    # rf_pred = rf.predict(X_test)

    # print(classification_report(y_test, rf_pred))
    # rf_cm = confusion_matrix(y_test, rf_pred, labels=sorted(list(set(y_test))))
    # print(rf_cm)
    # draw.draw_confusion_matrix(y_test, rf_pred, label_name=[str(l) for l in sorted(set(y_test))], title="sklearn RF Confusion Matrix", pdf_save_path="sklearn_rf_confusion_matrix.png", dpi=300)
    
#前向选择
    # start_time = time.time()
    
    # max_features = int(X_train.shape[1] * 0.5)
    # best_features, scores = fs.forward_feature_selection(X_train, y_train, X_test, y_test, max_features=max_features, step=5)
    # print('前向选择最佳特征:', best_features)
    # print('每步得分:', scores)

    # end_time = time.time()
    # print(f"特征选择与评估耗时: {end_time - start_time:.2f} 秒")

    best_features = ['1', '8', '9', '10','11', '12',  '14', '15', '16','20']  

    # gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # gbdt.fit(X_train[best_features], y_train,print_debug=True)
    # y_pred = gbdt.predict(X_test[best_features])

    # gbdt.fit(X_train, y_train,print_debug=True)
    # y_pred = gbdt.predict(X_test)

    # print(classification_report(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred, labels=sorted(list(set(y_test))))
    # print(cm)
    # draw.draw_confusion_matrix(y_test, y_pred, label_name=[str(l) for l in sorted(set(y_test))], title="sklearn GBDT Confusion Matrix (Forward Selection)", pdf_save_path="sklearn_gbdt_confusion_matrix_forward.png", dpi=300)


    le = LabelEncoder()
    Y_encode = le.fit_transform(y_train)
    # class_w = {
    #     'normal': 0.1,
    #     'dos': 0.6,
    #     'probe': 0.6,
    #     'r2l': 2,
    #     'u2r': 1.2
    # }

    class_w = {
        '1': 0.1,
        '0': 0.6,
        '2': 0.6,
        '3': 2,
        '4': 1.2
    }
    
    # class_w = {
    #     '1': 1,
    #     '0': 1,
    #     '2': 1,
    #     '3': 1,
    #     '4': 1
    # }

    # class_w = {
    #     '0': 0.1,
    #     '1': 1.2,
    #     '2': 1.2,
    #     '3': 0.8,
    #     '4': 1.2,
    #     '5': 1,
    #     '6': 0.8,
    #     '7': 1,
    #     '8': 1.2,
    #     '9': 0.8
    # }

    if hasattr(le, 'inverse_transform'):
        class_w_num = {le.transform([k])[0]: v for k, v in class_w.items() if k in le.classes_}
    else:
        class_w_num = class_w
    sample_w = compute_sample_weight(class_weight=class_w_num, y=Y_encode)
    dtrain = lgb.Dataset(X_train[best_features].values, label=Y_encode, weight=sample_w)
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(Y_encode)),
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'verbose': -1
    }
    train_start_time = time.time()
    lgb_model = lgb.train(params, dtrain, num_boost_round=100)
    train_end_time = time.time()
    print(f"训练耗时: {train_end_time - train_start_time:.2f} 秒")

    Y_pred_prob = lgb_model.predict(X_test[best_features].values)
    y_pred = np.argmax(Y_pred_prob, axis=1)
    y_pred_lgb = le.inverse_transform(y_pred)
    print('classification_report:')
    print(classification_report(y_test, y_pred_lgb))
    cm_lgb = confusion_matrix(y_test, y_pred_lgb, labels=sorted(list(set(y_test))))
    print(cm_lgb)


# 绘制混淆矩阵
    
    # label_names = ['dos','normal', 'probe','r2l','u2r']
    # cm_lgb_norm = cm_lgb.astype('float') / cm_lgb.sum(axis=1, keepdims=True)
    # fig, ax = plt.subplots(figsize=(6,6))
    # im = ax.imshow(cm_lgb_norm, cmap='Blues')
    # ax.set_xlabel('预测标签', fontsize=20)
    # ax.set_ylabel('真实标签', fontsize=20)
    # ax.set_xticks(np.arange(len(label_names)))
    # ax.set_yticks(np.arange(len(label_names)))
    # ax.set_xticklabels(label_names, rotation=45, fontsize=20)
    # ax.set_yticklabels(label_names, fontsize=20)
    # plt.title('LightGBM Confusion Matrix', fontsize=20)
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # for i in range(len(label_names)):
    #     for j in range(len(label_names)):
    #         color = 'white' if i == j else 'black'
    #         value = cm_lgb_norm[i, j]
    #         ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=15)
    # plt.tight_layout(rect=[0, 0, 0.92, 1])
    # plt.savefig('lgbm_confusion_matrix.png', dpi=300)
    # plt.close()

    result_df = pd.DataFrame({
        'True Label': y_test.values,
        'Predicted Label': y_pred,
        'Match': (y_test.values == y_pred).astype(int)
    })
    result_df.to_csv('./result.csv', index=False)
    print("预测结果已保存到 ./result.csv")


    # class_w = {
    #     '1': 0.1,
    #     '0': 0.6,
    #     '2': 0.6,
    #     '3': 2,
    #     '4': 1.2
    # }

    # gbdt_model = GBDTMultiClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,task='multiclass')
    # sample_w = fs.compute_sample_weight(class_weight=class_w, y=y_train)
    # gbdt_model.fit(X_train[best_features].values, y_train.values, sample_weight=sample_w)
    # test_predictions = gbdt_model.predict(X_test[best_features].values)

    # print(classification_report(y_test, test_predictions))

    
    # result_df = pd.DataFrame({
    #     "True Label": y_test,
    #     "Predicted Label": test_predictions,
    #     "Match": (y_test == test_predictions).astype(int)
    # })
    # result_df.to_csv("./result.csv", index=False)
    # print("Results saved to './result.csv'")

if __name__ == "__main__":
    main()
