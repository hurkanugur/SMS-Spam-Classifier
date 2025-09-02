import torch
from config import SPAM_THRESHOLD
from model_utils import SpamClassifier, load_model, load_vectorizer

def main():
    # Load model and vectorizer
    vectorizer = load_vectorizer()
    input_dim=len(vectorizer.get_feature_names_out())
    model = load_model(SpamClassifier, input_dim=input_dim)
    
    message = "Congratulations! You’ve won a free vacation and a $1,000 prize check. Call now!"
    X_real = torch.tensor(
        vectorizer.transform([message]).todense(),
        dtype=torch.float32
    )

    # Model inference
    with torch.no_grad():
        y_pred = model(X_real)
        probability = torch.sigmoid(y_pred).item()
        prediction = "SPAM" if probability > SPAM_THRESHOLD else "NOT SPAM"
            
    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction} (prob={probability:.4f})")

if __name__ == "__main__":
    main()
