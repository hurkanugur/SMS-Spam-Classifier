import torch
from torch import nn
from config import (
     MODEL_PATH, SPAM_THRESHOLD, TRAINING_EPOCHS, VAL_INTERVAL, VECTORIZER_PATH, LEARNING_RATE
)
from data_utils import extract_features_and_target, load_raw_data, split_data
from model_utils import SpamClassifier, save_model, save_vectorizer
from plot_utils import LossPlotter

def train_model(model, X_train, y_train, loss_fn, optimizer, X_val, y_val, loss_plotter: LossPlotter):
    """
    Train and monitor the model.
    """
        
    for epoch in range(TRAINING_EPOCHS):
        model.train()                       # Set model to training mode
        optimizer.zero_grad()               # Clear old gradients

        y_pred = model(X_train)             # Forward pass
        loss = loss_fn(y_pred, y_train)     # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update parameters

        # Record losses
        if (epoch % VAL_INTERVAL == 0) or (epoch == TRAINING_EPOCHS - 1):
            train_loss = loss.item()
            val_loss = validate_model(model, X_val, y_val, loss_fn).item()
            
            # Update live plot
            loss_plotter.update(train_loss=train_loss, val_loss=val_loss)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

    return model


def validate_model(model, X_val, y_val, loss_fn):
    """
    Validate the model.
    """

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)

    return loss

def test_model(model, X_data, y_data):
    """
    Test the model.
    """
        
    model.eval()
    with torch.no_grad():
        logits = model(X_data)
        y_pred = torch.sigmoid(logits) > SPAM_THRESHOLD
        accuracy = (y_pred == y_data).float().mean()
        sensitivity = (y_pred[y_data == 1] == y_data[y_data == 1]).float().mean()
        specificity = (y_pred[y_data == 0] == y_data[y_data == 0]).float().mean()
        precision = (y_pred[y_pred == 1] == y_data[y_pred == 1]).float().mean()
    
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")

def main():

    # Load raw data
    df = load_raw_data()

    # Extract features and target
    X, y, vectorizer = extract_features_and_target(df)
    
    # Split data (training/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Model, loss, optimizer
    model = SpamClassifier(input_dim=X.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Real-time loss plotter initialization
    loss_plotter = LossPlotter()

    # Train with validation monitoring
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        X_val=X_val,
        y_val=y_val,
        loss_plotter=loss_plotter
    )

    # Test the model
    test_model(model, X_test, y_test)

    # Save model + vectorizer
    save_model(model)
    save_vectorizer(vectorizer)

    # Keep the final plot displayed
    loss_plotter.close()

if __name__ == "__main__":
    main()