import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MAX_VOCAB_SIZE, SPLIT_DATASET, SPLIT_RANDOM_STATE, TEST_SET_SIZE, TRAIN_SET_SIZE, VAL_SET_SIZE

def load_raw_data():
    """
    Load the raw SMS spam dataset.
    """
        
    df = pd.read_csv(DATA_PATH, sep="\t", names=["type", "message"])
    df["spam"] = (df["type"] == "spam")
    df.drop("type", axis=1, inplace=True)

    print(f"\n• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(f"{df.head()}\n")

    return df

def extract_features_and_target(df):
    """
    Extracts input features (X) and target values (y) from the dataset.
    """
        
    vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE)
    X = torch.tensor(vectorizer.fit_transform(df["message"]).todense(), dtype=torch.float32)
    y = torch.tensor(df["spam"].values, dtype=torch.float32).reshape((-1, 1))
    return X, y, vectorizer

def split_data(X, y):
    """
    Split raw data into train/val/test sets.
    """

    if not SPLIT_DATASET:
        print("• Dataset splitting disabled. Using the same dataset for train/val/test.\n")
        return X, X, X, y, y, y
        
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_SET_SIZE), stratify=y, random_state=SPLIT_RANDOM_STATE
    )
    relative_test_size = TEST_SET_SIZE / (VAL_SET_SIZE + TEST_SET_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=SPLIT_RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
