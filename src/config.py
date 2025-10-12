# -------------------------
# Paths
# -------------------------
MODEL_PATH = "model/sms_spam_classifier.pth"
VECTORIZER_PATH = "model/vectorizer.pkl"
DATASET_CSV_PATH = "data/SMSSpamCollection"

# -------------------------
# Vocabulary
# -------------------------
VOCABULARY_SIZE = 1000

# -------------------------
# Training Hyperparameters
# -------------------------
LEAKY_RELU_SLOPE = 0.01
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 256
NUM_EPOCHS = 250
VAL_INTERVAL = 5

# -------------------------
# Dataset Splits
# -------------------------
SPLIT_DATASET = True
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
SPLIT_RANDOMIZATION_SEED = 42   # Int -> Reproducible splits | None -> Fully random splits

# -------------------------
# Prediction / Classification
# -------------------------
CLASSIFICATION_THRESHOLD = 0.5