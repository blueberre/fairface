# FairFace: Fairness-Aware Gender and Race Classification using Multi-Task Learning

This project trains a **multi-task deep learning model** to predict **gender** and **race** from facial images using the [FairFace](https://arxiv.org/abs/1908.04913) dataset. It also evaluates the **fairness** of gender classification across racial subgroups by computing metrics like **False Positive Rate (FPR)** and **False Negative Rate (FNR)** per race.

---

## 📌 Project Objective

Facial classification systems often exhibit **biases** against underrepresented groups. This project aims to:

- Train a multi-task model for **gender and race classification**
- Use **ResNet-18** as a feature extractor
- Evaluate **subgroup fairness** in gender predictions across racial categories
- Provide a clean and reproducible ML pipeline with fairness evaluation

---

## 🗂️ Project Structure

├── data/

  └── fairface_label_subset.csv # Preprocessed subset of FairFace labels

  ├── fairface_label_train.csv # Original label CSV
  
  ├── fairface_label_val.csv # Original label CSV

├── datasets/

  └── fairface_dataset.py # PyTorch dataset class

├── models/

  └── multitask_model.py # Multi-task ResNet-18 model

├── checkpoints/

  └── multitask_model.pth # Trained model weights (not uploaded)

├── fairface-imgs/ # Image folder (not uploaded)

├── preprocess.py # Label preprocessing and encoding

├── train.py # Training loop

├── evaluate.py # Group fairness evaluation

├── requirements.txt

└── README.md


---

## 🚀 How to Run the Project

> ⚠️ **Note:** The image dataset `fairface-imgs/` and model checkpoint `.pth` are **not uploaded** due to size. Please download the [FairFace images](https://github.com/joojs/fairface) and place them in the correct folder.

### 1. 🔧 Setup Environment

```
# Install dependencies
pip install -r requirements.txt
```

### 2. 🧹 Preprocess Labels

```
python preprocess.py
```

This will generate data/fairface_label_subset.csv with:

- Normalized gender/race
- Age group categorization (young/old)
- Numeric labels
- Full image file paths

### 3. 🏋️‍♂️ Train the Model

```
python train.py
```

- Trains a multi-task ResNet-18 model
- Predicts both gender (2 classes) and race (7 classes)
- Saves weights to checkpoints/multitask_model.pth

### 4. 📈 Evaluate Fairness

```
python evaluate.py
```

This evaluates gender prediction performance by race group which helps uncover demographic disparities in model performance.

---

## 📊 Interpretation of Results

The evaluation computes:

- Accuracy: Overall correctness
- FPR (False Positive Rate): % of males wrongly predicted as female
- FNR (False Negative Rate): % of females wrongly predicted as male

Disparities in FPR/FNR across racial groups indicate potential biases. For example:

- A higher FNR for a particular race may mean the model misclassifies females more often from that race.

---

## 📚 Citations

- **FairFace Dataset**  
  > Karkkainen, K., & Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age*. CVPR Workshop on Fairness in AI.  
  > [https://arxiv.org/abs/1908.04913](https://arxiv.org/abs/1908.04913)

