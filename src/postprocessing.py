"""
Author: Annam.ai IIT Ropar
Team Members: Aman Sagar
Leaderboard Rank: 16

"""



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_preds.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_preds.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()



!pip install -q timm

import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)

best_val_acc = 0
patience, trigger = 3, 0

EPOCHS = 20
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print("Early stopping.")
            break


from torchvision.transforms import functional as TF
import numpy as np

model.eval()
all_preds = []
all_img_ids = []

tta_transforms = [
    lambda x: x,
    lambda x: TF.hflip(x),
    lambda x: TF.vflip(x),
    lambda x: TF.rotate(x, 15),
]

with torch.no_grad():
    for inputs, img_ids in test_loader:
        batch_preds = []
        for tform in tta_transforms:
            augmented = torch.stack([tform(img.cpu()) for img in inputs])
            augmented = augmented.to(device)
            outputs = model(augmented)
            preds = torch.softmax(outputs, dim=1)
            batch_preds.append(preds.cpu().numpy())
        mean_preds = np.mean(batch_preds, axis=0)
        final_preds = np.argmax(mean_preds, axis=1)
        all_preds.extend(final_preds)
        all_img_ids.extend(img_ids)

# Reverse label_map to get soil_type string from label index
inv_label_map = {v: k for k, v in label_map.items()}
predicted_soil_types = [inv_label_map[p] for p in all_preds]

import pandas as pd
submission = pd.DataFrame({
    "image_id": all_img_ids,
    "soil_type": predicted_soil_types
})

submission.to_csv("submission5.csv", index=False)
print("Submission file created: submission5.csv")

from google.colab import files
files.download("submission5.csv")




