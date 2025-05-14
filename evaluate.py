import pandas as pd
import draw


    # Load attack types
    # with open("data\\NSL-KDD-DataSet\\attack_types.txt", "r") as f:
    #     attack_types = [line.strip() for line in f.readlines()]

    # Correct label_mapping and reverse_label_mapping
    # label_mapping = {
    #     0: 'Normal', 1: 'Backdoor', 2: 'Analysis', 3: 'Fuzzers', 
    #     4: 'Shel', 6: 'Exploits', 7: 'DoS', 8: 'Worms', 9: 'Generic'
    # }

def score(result_df):

    # Convert numeric labels to attack types
    result_df["True Label"] = result_df["True Label"].map(label_mapping)
    result_df["Predicted Label"] = result_df["Predicted Label"].map(label_mapping)

    # Calculate accuracy
    accuracy = (result_df["True Label"] == result_df["Predicted Label"]).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate precision
    true_positive = result_df[result_df["True Label"] == result_df["Predicted Label"]].groupby("Predicted Label").size()
    predicted_positive = result_df.groupby("Predicted Label").size()
    precision = (true_positive / predicted_positive).fillna(0)

    # Calculate recall for each attack type
    actual_positive = result_df.groupby("True Label").size()
    recall = (true_positive / actual_positive).fillna(0)

    # Display results
    print("\nPrecision by Attack Type:")
    print(precision)

    print("\nRecall by Attack Type:")
    print(recall)


    # 计算二分类（1与非1）下的指标
    binary_true = (result_df["True Label"] == 'normal').astype(int)
    binary_pred = (result_df["Predicted Label"] == 'normal').astype(int)

    print("\nbinary_true前10:", binary_true.head(10).tolist())
    print("binary_pred前10:", binary_pred.head(10).tolist())

    TP = ((binary_true == 1) & (binary_pred == 1)).sum()
    TN = ((binary_true != 1) & (binary_pred != 1)).sum()
    FP = ((binary_true != 1) & (binary_pred != 1)).sum()
    FN = ((binary_true == 1) & (binary_pred != 1)).sum()

    print(f"\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    binary_accuracy = (TP + TN) / (TP + TN + FP + FN)
    binary_precision = TP / (TP + FP) if (TP + FP) > 0 else -1
    binary_recall = TP / (TP + FN) if (TP + FN) > 0 else -1
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else -1

    print("\n[二分类(1与非1)指标]")
    print(f"Accuracy: {binary_accuracy:.4f}")
    print(f"Precision: {binary_precision:.4f}")
    print(f"Recall: {binary_recall:.4f}")
    print(f"F1-score: {binary_f1:.4f}")

if __name__ == "__main__":
    reverse_label_mapping = {'dos': 0, 'normal': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
    label_mapping = {attack: i for i, attack in reverse_label_mapping.items()}

    
    result_df = pd.read_csv("./result.csv")
    score(result_df)

    # huffman_result_df = pd.read_csv("./kdd_huffman/result.csv")
    # onehot_result_df = pd.read_csv("./kdd_onehot/result.csv")
    # score(huffman_result_df)
    # score(onehot_result_df)


    label_names = sorted(list(set(result_df["True Label"]).union(set(result_df["Predicted Label"]))))
    draw.draw_confusion_matrix(
        label_true=result_df["True Label"],
        label_pred=result_df["Predicted Label"],
        label_name=label_names,
        title="Confusion Matrix",
        pdf_save_path="confusion_matrix.png",
        dpi=300
    )
    print("Confusion matrix saved to 'confusion_matrix.png'")

    # draw.draw_time_cost_bar_chart(
    #     fit_time_df=pd.read_csv("./gbdt_train_time.csv"),
    #     title="Time Cost of Each Tree",
    #     pdf_save_path="time_cost.png",
    #     dpi=300
    # )
    # print("Time cost chart saved to 'time_cost.png'")

    # huffman_time_df = pd.read_csv("./kdd_huffman/gbdt_train_time.csv")
    # onehot_time_df = pd.read_csv("./kdd_onehot/gbdt_train_time.csv")
    # draw.draw_compare_train_time_line_chart(
    #     huffman_time_df=huffman_time_df,
    #     onehot_time_df=onehot_time_df,
    #     pdf_save_path="train_time_comparison.png",
    #     dpi=300
    # )
    # print("Train time comparison chart saved to 'train_time_comparison.png'")


