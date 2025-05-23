import pandas as pd

from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,f1_score

import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_sample_weight

# 分离离散变量
def split_category(data, columns):
    cat_data = data[columns]
    rest_data = data.drop(columns, axis=1)
    return rest_data, cat_data
#  转所有离散变量为one-hot
def one_hot_cat(data):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[data.name])
    out = pd.DataFrame([])
    for col in data.columns:
        one_hot_cols = pd.get_dummies(data[col], prefix=col)
        out = pd.concat([out, one_hot_cols], axis=1)
    out.set_index(data.index)
    return out

def main():
    '''
    # train_file = './data/UNSW_NB15_training-set.csv'
    # test_file = './data/UNSW_NB15_testing-set.csv'
    
    train_file = './data/KDDTrain+.csv'
    test_file = './data/KDDTest+.csv'
    attack_type_file = './data_2/attack_types.txt'
    field_name_file = './data_2/Field Names.csv'

    df = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    field_names_df = pd.read_csv(
        field_name_file, header=None, names=[
            'name', 'data_type']) # 定义dataframe ，并给个column name，方便索引
    field_names = field_names_df['name'].tolist()
    field_names += ['label', 'label_code'] # 源文件中没有标签名称，以及等级信息
    df = pd.read_csv(train_file, header=None, names=field_names)
    df_test = pd.read_csv(test_file, header=None, names=field_names)
    attack_type_df = pd.read_csv(
        attack_type_file, sep=' ', header=None, names=[
            'name', 'attack_type'])
    attack_type_dict = dict(
        zip(attack_type_df['name'].tolist(), attack_type_df['attack_type'].tolist())) # 定义5大类和小类的映射字典，方便替代
    df.drop('label_code', axis=1, inplace=True) # 最后一列 既无法作为feature，也不是我们的label，删掉
    df_test.drop('label_code', axis=1, inplace=True)
    df['label'].replace(attack_type_dict, inplace=True) # 替换label 为5 大类
    df_test['label'].replace(attack_type_dict, inplace=True)
    
    df.to_csv('processed_train.csv', index=False)
    df_test.to_csv('processed_test.csv', index=False)

    '''
    df = pd.read_csv("processed_train.csv")
    df_test = pd.read_csv("processed_test.csv")

    Y = df.iloc[:, -1]
    Y_test = df_test.iloc[:, -1]
    X = df.iloc[:, 1:-1]
    X_test = df_test.iloc[:, 1:-1]


    train_label= df[['label']]
    test_label= df_test[['label']]

    # train_label= df[['attack_cat']]
    # test_label= df_test[['attack_cat']]

#画画
    # train_label['type'] = 'train'
    # test_label['type'] = 'test'
    # label_all = pd.concat([train_label,test_label],axis=0)
    # print(label_all)
    # print(test_label)
    # plt.figure(figsize=(10, 6))  # 设置图的大小
    # ax = sns.countplot(x='label',hue='type', data=label_all)
    # # ax = sns.countplot(x='attack_cat',hue='type', data=label_all)
    # for p in ax.patches:
    #     height = p.get_height()
    #     if height > 0:
    #         ax.annotate(f'{int(height)}',
    #                     (p.get_x() + p.get_width() / 2, height),
    #                     ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 2), textcoords='offset points')
    # plt.xticks(rotation=45)
    # plt.subplots_adjust(bottom=0.18)  # 调整x轴标签距离底部的距离
    # plt.show()
#画画


    # categorical_columns
    categorical_mask = (X.dtypes == object)
    categorical_columns = X.columns[categorical_mask].tolist()

#重采样
    def label_encoder(data):
        labelencoder = LabelEncoder()
        for col in data.columns:
            data.loc[:,col] = labelencoder.fit_transform(data[col])
        return data
    # first label_encoder to allow resampling
    X[categorical_columns] = label_encoder(X[categorical_columns])
    X_test[categorical_columns] = label_encoder(X_test[categorical_columns])

    oversample = ADASYN()
    X, Y = oversample.fit_resample(X, Y)

    # print(f"Resampled X shape: {X.shape}, Y shape: {Y.shape}")
    # # 保存重采样后的前五行到csv
    # # 输出所有字符串列到文件
    # str_cols = X.select_dtypes(include=['object']).columns.tolist()
    # X_str = X[str_cols]
    # X_str.to_csv('resampled_string_columns.csv', index=False)

# def emt():

# convert to one-hot
    '''
    X, X_cat = split_category(X, categorical_columns)
    X_test, X_test_cat = split_category(X_test, categorical_columns)

    X_cat_one_hot_cols = one_hot_cat(X_cat)
    X_test_cat_one_hot_cols = one_hot_cat(X_test_cat)
    # align train to test
    X_cat_one_hot_cols, X_test_cat_one_hot_cols = X_cat_one_hot_cols.align(
        X_test_cat_one_hot_cols, join='inner', axis=1)
    X_cat_one_hot_cols.fillna(0, inplace=True)
    X_test_cat_one_hot_cols.fillna(0, inplace=True)
    X = pd.concat([X, X_cat_one_hot_cols], axis=1)
    X_test = pd.concat([X_test, X_test_cat_one_hot_cols],
                    axis=1)
    print(f'add one-hot features')
    print(f'x shape is {X.shape}')
    
    feature_name = list(X.columns) # 特征名称后续会用到
    print(feature_name)
    '''
    
    Y_encode = LabelEncoder().fit_transform(Y)
    Y_test_encode = LabelEncoder().fit_transform(Y_test)
    


    class_w = {
        'normal': 0.1,  # 0.1
        'dos': 0.6,
        'probe': 0.6,
        'r2l': 5,
        'u2r': 8} #以上数据需要微调，调整一般从normal开始，因为它的权重大
    from sklearn.utils.class_weight import compute_sample_weight
    sample_w = compute_sample_weight(class_weight=class_w, y=Y)
    ##!!然后传入该权重到数据集中
    dtrain = lgb.Dataset(X.values, label=Y_encode,weight=sample_w)

    # dtrain = lgb.Dataset(X.values, label=Y_encode)
    dtest = lgb.Dataset(X_test.values, label=Y_test_encode)
    param = {
        'eta': 0.1,
        'objective': 'multiclass',
        'num_class': 5,
        'verbose': 0,
            'metric':'multi_error'
    } # 参数几乎都是默认值，仅仅修改一些多分类必须的参数
    evals_result = {}
    valid_sets = [dtrain, dtest]
    valid_name = ['train', 'eval']

    model = lgb.train(param, dtrain, num_boost_round=100, 
                    valid_sets=valid_sets, valid_names=valid_name)

    y_pred_1 = model.predict(X_test.values)

    y_pred = pd.DataFrame(y_pred_1).idxmax(axis=1) #预测概率值转为预测标签
    #
    # 我们用了多种metric 来衡量结果，其中有些是明显不适合的，比如accuracy，因为它会被不平衡的数据分布带到阴沟里（误导）。
    print(f'auc score is {accuracy_score(Y_test_encode, y_pred)}')
    print(confusion_matrix(Y_test_encode, y_pred))
    print(classification_report(Y_test_encode, y_pred, digits=3))

    auc = roc_auc_score(Y_test_encode, y_pred_1, multi_class="ovo", average="macro") # 选用macro 很重要。参考sklearn。
    #Calculate metrics for each label, and find their unweighted mean. #This does not take label imbalance into account.
    print(f'roc_auc_score  is {auc}')

    f1 = f1_score(y_pred, Y_test_encode, average='macro')
    print(f'f1_score  is {f1}')


if __name__ == "__main__":
    main()
