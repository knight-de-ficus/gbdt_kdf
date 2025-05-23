import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from IPython.core.display import display, HTML
import plotly.graph_objects as go

from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder

def forward_feature_selection(X_train, y_train, X_test, y_test, max_features=10, step=1):
    """
    前向特征选择
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score
    import numpy as np

    selected = []
    remaining = list(X_train.columns)
    best_score = 0
    best_features = []

    for round_idx in range(min(max_features // step, len(remaining) // step + 1)):
        print(f"round {round_idx + 1} / {max_features // step}, size: {len(remaining)}")
        best_this_round = []
        candidates = []
        for feat in remaining:
            feats = selected + [feat]
            clf = GradientBoostingClassifier(n_estimators=3, learning_rate=0.1, random_state=42)
            clf.fit(X_train[feats], y_train)
            y_pred = clf.predict(X_test[feats])
            score = f1_score(y_test, y_pred, average='macro')
            candidates.append((feat, score))
            print("#",end="",flush=True)

        # select step bset
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_this_round = [feat for feat, _ in candidates[:step] if feat in remaining]
        if best_this_round:
            selected.extend(best_this_round)
            for feat in best_this_round:
                remaining.remove(feat)
            # 用当前所有已选特征评估分数
            # clf = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, random_state=42)
            # clf.fit(X_train[selected], y_train)
            # y_pred = clf.predict(X_test[selected])
            # best_score = f1_score(y_test, y_pred, average='macro')
            best_features = selected.copy()
            # scores.append(best_score)
        else:
            break
        print("")
    return best_features

def chi2_select_features(train_file, test_file, keep_ratio=0.8, show_and_save_top_features=True, save_path="selected_features.png"):

    from sklearn.feature_selection import SelectKBest, chi2
    import matplotlib.pyplot as plt
    X = train_file.iloc[:, 1:-1]
    y = train_file.iloc[:, -1]
    n_features = X.shape[1]
    k = int(n_features * keep_ratio)
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_columns = X.columns[mask]
    
    keep_cols = [train_file.columns[0]] + list(selected_columns) + [train_file.columns[-1]]
    train_selected = train_file[keep_cols].copy()
    test_selected = test_file[keep_cols].copy()

    if show_and_save_top_features:
        
        scores = selector.scores_[mask]
        feature_score = pd.DataFrame({'feature': selected_columns, 'score': scores})
        feature_score = feature_score.sort_values(by='score', ascending=False)
        plt.figure(figsize=(10, min(0.5 * k, 20)))
        plt.barh(feature_score['feature'], feature_score['score'], color='skyblue')
        plt.xlabel('Score', fontsize=16)
        plt.ylabel('Feature', fontsize=16)
        plt.title(f'Top {k} Features', fontsize=18)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        # plt.show()


    return train_selected, test_selected

def chi2_draw(df,save_path="selected_features.png"):
    best_features = SelectKBest(score_func=chi2,k='all')

    X = df.iloc[:,1:-2]
    y = df.iloc[:,-1]
    fit = best_features.fit(X,y)

    df_scores=pd.DataFrame(fit.scores_)
    df_col=pd.DataFrame(X.columns)

    feature_score=pd.concat([df_col,df_scores],axis=1)
    feature_score.columns=['feature','score']
    feature_score.sort_values(by=['score'],ascending=True,inplace=True)

    fig = go.Figure(go.Bar(
                x=feature_score['score'][0:21],
                y=feature_score['feature'][0:21],
                orientation='h'))

    fig.update_layout(title="Top 20 Features",
                      height=1200,
                      showlegend=False,
                     )

    fig.write_image(save_path)
    # fig.show()

def compute_sample_weight(class_weight, y):
    """
    样本的权重
    """
    y = np.array(y)
    weights = np.ones_like(y, dtype=float)
    for label, w in class_weight.items():
        weights[y == label] = w
    return weights

def label_encoder(data):
    labelencoder = LabelEncoder()
    for col in data.columns:
        data.loc[:, col] = labelencoder.fit_transform(data[col])
    return data

def oversample_data(train_df, categorical_columns=None):
    """
    过采样
    0.5倍
    """
    X = train_df.iloc[:, 1:-1].copy()
    y = train_df.iloc[:, -1].copy()
    if categorical_columns is not None and len(categorical_columns) > 0:
        X[categorical_columns] = label_encoder(X[categorical_columns])
    # 统计主类数量
    value_counts = y.value_counts()
    max_count = value_counts.max()
    # 构造采样策略
    sampling_strategy = {cls: int(0.5 * max_count) for cls in value_counts.index if value_counts[cls] < int(0.5 * max_count)}
    oversample = ADASYN(sampling_strategy=sampling_strategy)
    X_res, y_res = oversample.fit_resample(X, y)
    id_col = train_df.columns[0]
    new_train_df = pd.DataFrame(X_res, columns=X.columns)
    new_train_df.insert(0, id_col, range(len(new_train_df)))
    new_train_df[train_df.columns[-1]] = y_res
    return new_train_df
