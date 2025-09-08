import os
import pickle
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import config


class SMSDataset(Dataset):
    """
    Dataset class for exam pass/fail classification.
    Provides training data (split + normalize) and inference data (normalize only).
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=config.VOCABULARY_SIZE)

    # ----------------- PUBLIC METHODS -----------------
    
    def prepare_data_for_training(self):
        """Load data, split train/val/test, fit scaler, normalize features, and return DataLoaders."""
        df = self._load_csv()
        X = torch.tensor(self.vectorizer.fit_transform(df["message"]).todense(), dtype=torch.float32)
        y = torch.tensor(df["spam"].values, dtype=torch.float32).reshape((-1, 1))

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        train_loader, val_loader, test_loader = self._create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)

        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, df: pd.DataFrame):
        """Prepare the data for inference."""
        X = torch.tensor(
            self.vectorizer.transform(df["message"]).todense(),
            dtype=torch.float32
        )

        return X

    def get_flattened_input_size(self, data_loader):
        """Return number of input features per sample after flattening (for MLPs)."""
        sample_X, _ = next(iter(data_loader))
        input_dim = sample_X[0].numel()
        print(f"• Input dimension: {input_dim}")
        return input_dim

    def save_vectorizer(self):
        """Save the vectorizer for inference."""
        os.makedirs(os.path.dirname(config.VECTORIZER_PATH), exist_ok=True)
        with open(config.VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"• The vectorizer saved: {config.VECTORIZER_PATH}")

    def load_vectorizer(self):
        """Load the vectorizer from file (must be called before inference)."""
        with open(config.VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"• The vectorizer loaded: {config.VECTORIZER_PATH}")

    # ----------------- PRIVATE METHODS -----------------
    
    def _load_csv(self):
        """Load CSV into a pandas DataFrame."""
        df = pd.read_csv(config.DATASET_CSV_PATH, sep="\t", names=["type", "message"])
        df["spam"] = (df["type"] == "spam")
        df.drop("type", axis=1, inplace=True)

        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    def _split_dataset(self, X, y):
        """Split dataset into training, validation, and test subsets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val/test.")
            return X, X, X, y, y, y
        
        dataset = TensorDataset(X, y)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=generator
        )

        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Return train, val, test DataLoaders."""
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
