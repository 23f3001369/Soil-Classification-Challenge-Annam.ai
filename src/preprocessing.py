"""

Author: Annam.ai IIT Ropar
Team Members: Aman Sagar
Leaderboard Rank: 16

"""

from google.colab import drive
drive.mount('/content/drive')


import os

dataset_path = '/content/drive/My Drive/soil_classification-2025'
os.listdir(dataset_path)

!pip install -q pandas scikit-learn torchvision

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import torch
print("PyTorch version:", torch.__version__)


ROOT_DIR = "/content/drive/MyDrive/soil_classification-2025"
TRAIN_CSV = os.path.join(ROOT_DIR, "train_labels.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test_ids.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")


label_map = {
    "Alluvial soil": 0,
    "Black Soil": 1,
    "Clay soil": 2,
    "Red soil": 3
}

df = pd.read_csv(TRAIN_CSV)
df['label'] = df['soil_type'].map(label_map)
# df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
df_train = df  # use the full training set


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, test=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, img_id
        else:
            label = self.df.iloc[idx]['label']
            return image, label


BATCH_SIZE = 32

train_dataset = SoilDataset(df_train, TRAIN_DIR, transform=train_transforms)
# val_dataset = SoilDataset(df_val, TRAIN_DIR, transform=val_transforms)
val_dataset = SoilDataset(df_val, TRAIN_DIR, transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


test_df = pd.read_csv(TEST_CSV)
test_dataset = SoilDataset(test_df, TEST_DIR, transform=val_transforms, test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm  # nice progress bars

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load pretrained ResNet50 and modify for 4 classes
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)
