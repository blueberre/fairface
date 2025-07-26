import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from models.multitask_model import FairFaceMultiTaskModel
from datasets.fairface_dataset import FairFaceDataset


BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
NUM_RACE_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/fairface_label_subset.csv")

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["race"], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = FairFaceDataset(train_df, transform=transform)
test_dataset = FairFaceDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = FairFaceMultiTaskModel(num_race_classes=NUM_RACE_CLASSES).to(DEVICE)

criterion_gender = nn.CrossEntropyLoss()
criterion_race = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_gender, correct_race = 0, 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)

    for images, gender_labels, race_labels, _ in loop:
        images = images.to(DEVICE)
        gender_labels = gender_labels.to(DEVICE)
        race_labels = race_labels.to(DEVICE)

        optimizer.zero_grad()

        gender_logits, race_logits = model(images)

        loss_gender = criterion_gender(gender_logits, gender_labels)
        loss_race = criterion_race(race_logits, race_labels)
        loss = loss_gender + loss_race

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred_gender = torch.max(gender_logits, 1)
        _, pred_race = torch.max(race_logits, 1)
        correct_gender += (pred_gender == gender_labels).sum().item()
        correct_race += (pred_race == race_labels).sum().item()
        total += gender_labels.size(0)

        loop.set_postfix({
            "Loss": f"{running_loss:.3f}",
            "Gender Acc": f"{100*correct_gender/total:.2f}%",
            "Race Acc": f"{100*correct_race/total:.2f}%"
        })

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/multitask_model.pth")
print("Model saved to checkpoints/multitask_model.pth")

    