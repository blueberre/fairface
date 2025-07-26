import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def preprocess(save_path="data/fairface_label_subset.csv", total_samples=10000):
    train_df = pd.read_csv("fairface_label_train.csv")
    val_df = pd.read_csv("fairface_label_val.csv")

    train_df["split"] = "train"
    val_df["split"] = "val"

    df = pd.concat([train_df, val_df], ignore_index=True)

    df["age"] = df["age"].replace({"Oct-19": "10-19", "03-Sep": "03-09"})

    def convert_age_group(age):
        return "young" if age in ["00-02", "03-09", "10-19", "20-29"] else "old"

    df["age_group"] = df["age"].apply(convert_age_group)

    df["gender"] = df["gender"].str.lower()
    df["race"] = df["race"].str.lower()

    df_subset = (
        df.groupby("race")
        .apply(lambda x: x.sample(frac=min(1.0, total_samples / len(df)), random_state=42))
        .reset_index(drop=True)
    )

    gender_encoder = LabelEncoder()
    race_encoder = LabelEncoder()

    df_subset["gender_label"] = gender_encoder.fit_transform(df_subset["gender"])
    df_subset["race_label"] = race_encoder.fit_transform(df_subset["race"])

    df_subset["file_path"] = df_subset["file"].apply(lambda x: os.path.join("fairface-imgs", x))

    os.makedirs("data", exist_ok=True)
    df_subset.to_csv(save_path, index=False)
    print("Saved:", save_path)

if __name__ == "__main__":
    preprocess()
