# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/sms_spam_classifier.pth"
VECTORIZER_PATH = "../model/vectorizer.pkl"
DATASET_CSV_PATH = "../data/SMSSpamCollection"

# -------------------------
# Vocabulary
# -------------------------
VOCABULARY_SIZE = 1000

# -------------------------
# Training Hyperparameters
# -------------------------
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 64
NUM_EPOCHS = 100
VAL_INTERVAL = 1

# -------------------------
# Dataset Splits
# -------------------------
SPLIT_DATASET = True  # Set False to use the full dataset for train/val/test
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
SPLIT_RANDOMIZATION_SEED = 42  # Seed (integer) for reproducible dataset splits; set to None for fully random splits

# -------------------------
# Prediction / Classification
# -------------------------
CLASSIFICATION_THRESHOLD = 0.5  # Probability threshold to classify Pass vs Fail (used during inference/evaluation only)