import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from IPython.core.display import display, HTML
import plotly.graph_objects as go

from sklearn.feature_selection import SelectKBest, chi2
    
def select_features(train_file, test_file, keep_ratio=0.8, show_and_save_top_features=True, save_path="selected_features.png"):

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
