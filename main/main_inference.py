from src.device_manager import DeviceManager
from src.dataset import SMSDataset
from src.model import SMSSpamClassifier
from src.inference import InferencePipeline


def main():
    print("-------------------------------------")
    device_manager = DeviceManager()
    device = device_manager.device

    # Load the dataset and vectorizer
    dataset = SMSDataset()
    dataset.load_vectorizer()

    # Load the model
    input_dim = dataset.get_input_dim()
    model = SMSSpamClassifier(input_dim=input_dim, device=device)
    model.load()

    # Build inference pipeline
    inference_pipeline = InferencePipeline(model, dataset, device)
    app = inference_pipeline.create_gradio_app()

    # Launch the app
    app.launch(share=True)

    print("-------------------------------------")
    device_manager.release_memory()
    print("-------------------------------------")


if __name__ == "__main__":
    main()
