from src.dataset import SMSDataset
from src.model import SMSSpamClassifier
from src.train import TrainingPipeline
from src.visualize import LossMonitor
from src.device_manager import DeviceManager

def main():

    # Select CUDA (GPU) / MPS (Mac) / CPU
    device_manager = DeviceManager()
    device = device_manager.device

    # Load and prepare data
    dataset = SMSDataset()
    train_loader, val_loader, test_loader = dataset.prepare_data_for_training()

    # Create a new model
    input_dim = dataset.get_input_dim()
    model = SMSSpamClassifier(input_dim=input_dim, device=device)

    # Loss Monitoring
    loss_monitor = LossMonitor()

    # Create the training pipeline
    training_pipeline = TrainingPipeline(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        device=device, 
        loss_monitor=loss_monitor
    )

    # Train the model
    training_pipeline.train()

    # Test the model
    training_pipeline.test()

    # Save the model
    model.save()
    dataset.save_vectorizer()

    # Keep the final plot displayed
    loss_monitor.close()

    # Release the memory
    device_manager.release_memory()

if __name__ == "__main__":
    main()
