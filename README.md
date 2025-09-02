# 📧 SMS Spam Classifier with PyTorch

## 📖 Overview
This project predicts whether a given SMS message is **spam or not** using a **logistic regression model** implemented in **PyTorch**.  
It covers the full pipeline from data preprocessing to model training, evaluation, and inference on real messages, including:

- 📊 **Logistic Regression Model** implemented with PyTorch  
- ⚖️ **Binary Cross-Entropy (BCE) Loss** for classification  
- 🧠 **Adam optimizer** for training  
- 🔀 **Train/Validation/Test split** for robust evaluation  
- 📈 **CountVectorizer feature extraction** for converting text to numerical features

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling & preprocessing  
- **scikit-learn** – dataset splitting  
- **matplotlib** – plotting loss curves
- **pickle** – saving/loading vectorizer and model

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
└── SMSSpamCollection           # Raw dataset

model/
├── spam_model.pth              # Trained model (after training)
└── vectorizer.pkl              # Fitted CountVectorizer (after training)

src/
├── config.py                   # Configurations (paths, hyperparameters, dataset split)
├── data_utils.py               # Data loading, preprocessing, feature extraction
├── model_utils.py              # Model definition and save/load utilities
├── plot_utils.py               # Loss plotting
├── predict_spam_messages.py    # Predict spam on new SMS messages
├── train_model.py              # Model training and evaluation

requirements.txt                # Python dependencies

```
---

## 📂 Train the Model
```bash
python train_model.py
```
or
```bash
python3 train_model.py
```

---

## 📂 Run Predictions on Real Data
```bash
python predict_spam_messages.py
```
or
```bash
python3 predict_spam_messages.py
```
