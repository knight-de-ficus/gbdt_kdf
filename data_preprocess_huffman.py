import pandas as pd
from collections import Counter
import heapq

def huffman_encoding(data):

    # Count
    frequency = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Huffman codes
    huffman_dict = {}
    for item in heap:
        for symbol, code in item[1:]:
            huffman_dict[symbol] = code

    max_depth = max(len(code) for code in huffman_dict.values())

    return huffman_dict, max_depth


def preprocess_data(training_data, test_data, mode="one-hot",lable_index=-1):
    

    training_labels = training_data.iloc[:, lable_index]
    training_features = training_data.iloc[:, 1:lable_index]
    test_labels = test_data.iloc[:, lable_index]
    test_features = test_data.iloc[:, 1:lable_index]
    
    training_features.columns = range(training_features.shape[1])
    test_features.columns = range(test_features.shape[1])

    types = {label: idx for idx, label in enumerate(training_labels.unique())}
    print(types)

    for col in training_features.columns:
        if training_features[col].dtype == 'object':
            if mode == "huffman":
                huffman_dict, depth = huffman_encoding(training_features[col])
                # Map
                training_data_column = training_features[col].map(huffman_dict)
                test_data_column = test_features[col].map(huffman_dict)

                for i in range(depth):
                    training_features[f"{i+col*100}"] = training_data_column.apply(lambda x: x[i] if i < len(x) else '0')
                    test_features[f"{i+col*100}"] = test_data_column.apply(lambda x: str(x)[i] if isinstance(x, str) and i < len(x) else '0')
                # Drop
                training_features.drop(columns=[col], inplace=True)
                test_features.drop(columns=[col], inplace=True)
                print(f"{depth}\n")
            elif mode == "one-hot":
                one_hot_encoded = pd.get_dummies(training_features[col], prefix=None)
                for i, cname in enumerate(one_hot_encoded.columns):
                    training_features[f"{i+col*100}"] = one_hot_encoded[cname]
                training_features.drop(columns=[col], inplace=True)

                one_hot_encoded_test = pd.get_dummies(test_features[col], prefix=None)
                for i, cname in enumerate(one_hot_encoded_test.columns):
                    test_features[f"{i+col*100}"] = one_hot_encoded_test[cname] if cname in one_hot_encoded_test else 0
                test_features.drop(columns=[col], inplace=True)

                test_features = test_features.reindex(columns=training_features.columns, fill_value=0)

    # Create label_mapping and reverse_label_mapping
    label_mapping = {i: attack for i, attack in enumerate(types)}
    reverse_label_mapping = {attack: i for i, attack in label_mapping.items()}

    # Map training_labels and test_labels using reverse_label_mapping
    if training_labels.dtype == 'object':
        training_labels = training_labels.map(reverse_label_mapping)
        test_labels = test_labels.map(reverse_label_mapping)

    # Convert all feature column names to numeric values (column indices)


    return training_features, training_labels, test_features, test_labels, label_mapping

if __name__ == "__main__":
    # training_file_path = "data/UNSW_NB15_training-set.csv"
    # test_file_path = "data/UNSW_NB15_testing-set.csv"

    training_file_path = "data_2/kddtrain.csv"
    test_file_path = "data_2/kddtest.csv"

    training_data = pd.read_csv(training_file_path)
    test_data = pd.read_csv(test_file_path)

    training_features, training_labels, test_features, test_labels, label_mapping = preprocess_data(training_data, test_data, mode="huffman",lable_index=-2)

    training_data = pd.concat([training_features, training_labels], axis=1)
    training_data.to_csv("./data_2/kddtrain_index.csv", index=False)
    training_data = pd.concat([ test_features, test_labels], axis=1)
    training_data.to_csv("./data_2/kddtest_index.csv", index=False)