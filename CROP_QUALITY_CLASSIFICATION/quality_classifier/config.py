from pathlib import Path

DATA_DIR = Path(r"d:\cv start\data\quality")

MODEL_NAME = 'mobilenet-v3'      # mobilenet-v3 | efficientnet-b0 | resnet-50
NUM_CLASSES = 2                   # bad, normal
CLASS_NAMES = ['bad', 'normal']   # будет перезаписан из папок

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224

DEVICE = 'cuda'

SAVE_DIR = Path('./models')
SAVE_DIR.mkdir(exist_ok=True)
