from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class DayNightDataset(Dataset):
    def __init__(self, root_day, root_night, transform=None):
        self.root_day = root_day
        self.root_night = root_night
        self.transform = transform

        self.day_images = os.listdir(root_day)
        self.night_images = os.listdir(root_night)
        self.length_dataset = max(len(self.day_images), len(self.night_images))
        self.day_len = len(self.day_images)
        self.night_len = len(self.night_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        day_img = self.day_images[index % self.day_len]
        night_img = self.night_images[index % self.night_len]

        day_path = os.path.join(self.root_day, day_img)
        night_path = os.path.join(self.root_night, night_img)

        day_img = np.array(Image.open(day_path).convert("RGB"))
        night_img = np.array(Image.open(night_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=day_img, image0=night_img)
            day_img = augmentations["image"]
            night_img = augmentations["image0"]

        return day_img, night_img