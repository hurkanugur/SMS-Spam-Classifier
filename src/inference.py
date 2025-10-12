import torch
import torch.nn.functional as F
import gradio as gr

from src.dataset import SMSDataset
from src.model import SMSSpamClassifier
import src.config as config


class InferencePipeline:
    """
    Handles model loading, text preprocessing, and SMS spam classification.
    """

    # ----------------- Initialization -----------------

    def __init__(
            self, 
            model: SMSSpamClassifier, 
            dataset: SMSDataset, 
            device: torch.device
        ):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    # ----------------- Public Methods -----------------

    def predict(self, text: str) -> str:
        """
        Predict if an SMS is spam or ham.
        Returns formatted prediction string.
        """
        if not text or text.strip() == "":
            return "⚠️ Please enter a message first!"

        # Preprocess text
        X = self.dataset.prepare_data_for_inference(text)
        X = X.to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.sigmoid(outputs)
            prediction = (probabilities > config.CLASSIFICATION_THRESHOLD).float()
            confidence = round(probabilities[0].item() * 100, 2)

        label = "❌ Spam" if prediction.item() == 1 else "✅ Ham"
        return f"{label} — {confidence}% confidence"

    def create_gradio_app(self) -> gr.Blocks:
        """
        Build and return the Gradio interface for interactive SMS classification.
        """
        with gr.Blocks(theme=gr.themes.Ocean(), title="SMS Spam Classifier") as app:
            gr.Markdown(
                """
                # ✉️ SMS Spam Classifier  
                Enter a message below to see if it is classified as **Spam** or **Ham**.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="📝 Enter SMS",
                        placeholder="Type or paste an SMS message here...",
                        lines=5
                    )
                    analyze_btn = gr.Button("🔍 Analyze SMS", variant="primary")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🧾 Prediction Result")
                    output_text = gr.Textbox(
                        label="Prediction",
                        placeholder="The model's prediction will appear here...",
                        interactive=False,
                        lines=2,
                        show_copy_button=True,
                    )

            # Button actions
            analyze_btn.click(fn=self.predict, inputs=text_input, outputs=output_text)
            clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[text_input, output_text])

            gr.Markdown(
                """
                ---
                💡 **Tip:** Use real SMS messages for accurate predictions.  
                📊 Model trained on the SMS Spam Collection Dataset.  

                ---
                👨‍💻 Developed by [Hürkan Uğur](https://github.com/hurkanugur)  
                🔗 Source Code: [SMS-Spam-Classifier](https://github.com/hurkanugur/SMS-Spam-Classifier)
                """
            )

        return app
