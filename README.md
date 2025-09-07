# 📧 SMS Spam Classifier with PyTorch

## 📖 Overview
This project predicts whether a given SMS message is **spam or not** using a **feedforward neural network** implemented in **PyTorch**.  
It covers the full pipeline from data preprocessing to model training, evaluation, and inference on real messages, including:

- 🧠 **Neural Network Model** implemented with PyTorch  
- ⚖️ **Binary Cross-Entropy (BCE) Loss** for binary classification  
- 🔄 **Adam optimizer** for training  
- 🔀 **Train/Validation/Test split** with mini-batches for robust evaluation  
- 📈 **CountVectorizer feature extraction** for converting text messages into numerical features  
- 💾 **Saving and loading vectorizer** for consistent preprocessing during inference 

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling  
- **matplotlib** – loss visualization  
- **pickle** – saving/loading vectorizer and trained model

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/SMS-Spam-Classifier.git
```

- Navigate to the `SMS-Spam-Classifier` directory
```bash
cd SMS-Spam-Classifier
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `SMS-Spam-Classifier/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── SMSSpamCollection               # Raw dataset

model/
└── sms_spam_classifier.pth         # Trained model (after training)

src/
├── config.py                       # Paths, hyperparameters, split ratios
├── dataset.py                      # Data loading & preprocessing
├── main_train.py                   # Training & model saving
├── main_inference.py               # Inference pipeline
├── model.py                        # Neural network definition
├── visualize.py                    # Training/validation plots

requirements.txt                    # Python dependencies
```

---

## 📂 Model Architecture

```bash
Input → Linear(64) → ReLU
      → Linear(32) → ReLU
      → Linear(1) → Sigmoid(Output)
```

---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
