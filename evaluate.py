import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from models.multitask_model import FairFaceMultiTaskModel
from datasets.fairface_dataset import FairFaceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/fairface_label_subset.csv")

from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, stratify=df["race"], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = FairFaceDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


model = FairFaceMultiTaskModel(num_race_classes=7).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/multitask_model.pth", map_location=DEVICE))
model.eval()


all_preds = []
all_labels = []
all_races = []

with torch.no_grad():
    for images, gender_labels, race_labels, race_names in test_loader:
        images = images.to(DEVICE)
        gender_labels = gender_labels.to(DEVICE)

        gender_logits, _ = model(images)
        _, gender_preds = torch.max(gender_logits, 1)

        all_preds.extend(gender_preds.cpu().numpy())
        all_labels.extend(gender_labels.cpu().numpy())
        all_races.extend(race_names)


results = defaultdict(dict)
unique_races = sorted(set(all_races))

for race in unique_races:
    indices = [i for i, r in enumerate(all_races) if r == race]
    y_true = [all_labels[i] for i in indices]
    y_pred = [all_preds[i] for i in indices]

    if len(set(y_true)) < 2:
        print(f"Skipping race '{race}' — only one class present.")
        continue

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        print(f"Incomplete confusion matrix for '{race}':\n{cm}")
        continue

    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)

    results[race]["accuracy"] = round(acc * 100, 2)
    results[race]["FPR"] = round(fpr * 100, 2)
    results[race]["FNR"] = round(fnr * 100, 2)


print("\nFairness Evaluation by Race (Gender Prediction)")
print("{:<20s} {:>10s} {:>10s} {:>10s}".format("Race", "Accuracy", "FPR", "FNR"))
print("-" * 52)
for race in results:
    acc = results[race]["accuracy"]
    fpr = results[race]["FPR"]
    fnr = results[race]["FNR"]
    print(f"{race:<20s} {acc:>10.2f} {fpr:>10.2f} {fnr:>10.2f}")
