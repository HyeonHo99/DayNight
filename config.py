import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "D:/DB/DayNight_DB/Aachen/Aachen_sampled/train"
VAL_DIR = "D:/DB/DayNight_DB/Aachen/Aachen_sampled/train"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.1
LAMBDA_CYCLE = 10
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_GEN_night = "./checkpoints/gen_night.pth"
CHECKPOINT_GEN_day = "./checkpoints/gen_day.pth"
CHECKPOINT_CRITIC_night = "./checkpoints/critic_night.pth"
CHECKPOINT_CRITIC_day = "./checkpoints/critic_day.pth"
IMG_SIZE = 512

transforms = A.Compose(
    [
        A.Resize(width=IMG_SIZE, height=IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)