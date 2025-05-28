import pandas as pd

if __name__ == "__main__":
    # train_file_path = "./data_2/KDDTrain+.csv"
    train_file_path = "./data_3/kddtrain_f.csv"

    df = pd.read_csv(train_file_path)

    unique_values = df.iloc[:, -1].unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}

    print(value_to_int)

