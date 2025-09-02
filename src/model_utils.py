import torch
from torch import nn
import os
import pickle

from config import MODEL_PATH, VECTORIZER_PATH

class SpamClassifier(nn.Module):
    """Logistic regression model for classifying spam messages."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def save_model(model):
    """Save model parameters to a file."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n• Model has been saved: {MODEL_PATH}")

def load_model(model_class, input_dim):
    """Load model parameters from a file."""
    model = model_class(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def save_vectorizer(vectorizer):
    """Save the vectorizer to a file."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"• The vectorizer have been saved: {VECTORIZER_PATH}")

def load_vectorizer():
    """Load the vectorizer from a file."""
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer
