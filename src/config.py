# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/spam_model.pth"
VECTORIZER_PATH = "../model/vectorizer.pkl"
DATA_PATH = "../data/SMSSpamCollection"

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 0.01               # Learning rate
TRAINING_EPOCHS = 100000           # Number of training epochs
VAL_INTERVAL = 1000                # Interval for validation loss evaluation
MAX_VOCAB_SIZE = 1000              # Maximum number of words to keep in vocabulary
SPAM_THRESHOLD = 0.25              # x > value will be considered as a spam SMS

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True  # Set to False to use the same dataset for train/val/test
SPLIT_RANDOM_STATE = 42 # Seed for reproducible shuffling; set to None for random behavior
TRAIN_SET_SIZE = 0.7
VAL_SET_SIZE = 0.15
TEST_SET_SIZE = 0.15