import numpy as np
import pandas as pd
import torch
import config
from dataset import SMSDataset
from model import SMSSpamClassifier

def main():
    # -------------------------
    # Select CPU or GPU
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # -------------------------
    # Load dataset and vectorizer
    # -------------------------
    dataset = SMSDataset()
    dataset.load_vectorizer()

    # -------------------------
    # Example real-world input
    # -------------------------
    df = pd.DataFrame([
        {
            "spam": True,
            "message": "Free entry in a weekly competition to win FA Cup final tickets on 21st May 2005. Text FA to 87121 to receive the entry question (standard text rates apply). Terms & Conditions apply. Only for 18+."
        },
        {
            "spam": False,
            "message": "Hey, are we still meeting for lunch today?"
        },
        {
            "spam": True,
            "message": "Congratulations! You have won a $1000 Walmart gift card. Call 123-456-7890 to claim it now."
        },
        {
            "spam": False,
            "message": "Okay, haha... just joking with you."
        },
        {
            "spam": False,
            "message": "Please review the attached report before tomorrow's meeting."
        }
    ])

    X = dataset.prepare_data_for_inference(df)
    input_dim = X[0].numel()

    # -------------------------
    # Load trained model
    # -------------------------
    model = SMSSpamClassifier(input_dim=input_dim, device=device)
    model.load()

    # -------------------------
    # Make predictions
    # -------------------------
    model.eval()
    X = X.to(device=device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()

    # -------------------------
    # Display results
    # -------------------------
    for i, (p, prob) in enumerate(zip(predictions, probabilities)):
        label = 'Spam' if p.item() == 1 else 'Ham'
        print(f"SMS {i+1}: Predicted: {label}, Probability: {prob.item():.2f}")


if __name__ == "__main__":
    main()
