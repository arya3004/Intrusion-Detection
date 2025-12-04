# Intrusion-Detection
---

# 🚨 Intrusion Detection System (IDS) — Machine Learning Project

A machine learning–based Intrusion Detection System that analyzes network traffic and classifies flows as **benign** or **malicious**.
This project is inspired by the open-source [cstub/ml-ids](https://github.com/cstub/ml-ids) repository and has been extended with custom preprocessing, training, and evaluation.

---

## 📌 Features

* 🔍 Detects malicious vs benign network flows
* 🧹 Automated preprocessing: cleaning, encoding, scaling
* 📊 Feature engineering and dataset split
* 🤖 Multiple ML models implemented
* 📈 Model evaluation with accuracy, precision, recall, F1-score
* 🧪 Test scripts included
* 🗂️ Clean, modular folder structure

---

## 📁 Project Structure

```
ml-ids/
│── data/                   # Optional: dataset files
│── ml_ids/                 # Source code
│   ├── config.py
│   ├── training.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│── models/                 # Saved trained models
│── tests/                  # Test scripts
│── notebooks/              # Exploratory notebooks
│── requirements.txt
│── README.md
```

## 🚀 Getting Started

### **1. Clone the repository**

```bash
git clone https://github.com/arya3004/Intrusion-Detection.git
cd Intrusion-Detection
```

### **2. Create a virtual environment**

```bash
python -m venv venv
```

### **3. Activate it**

```bash
venv\Scripts\activate   # Windows
```

### **4. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🧠 Machine Learning Model Used

The machine learning estimator created in this project follows a supervised approach and is trained using the Gradient Boosting algorithm. Employing the CatBoost library a binary classifier is created, capable of classifying network flows as either benign or malicious. The chosen parameters of the classifier and its performance metrics can be examined in the following notebook.

## model is trained and evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📊 Dataset

You can use **any network intrusion dataset**, for example:

* CICIDS-2018
* NSL-KDD
* UNSW-NB15
* Custom CSV network traffic

Place your dataset in the **data/** folder.

---

## ▶️ Train the Model

```bash
python -m ml_ids.training
```

This script will:

* Load the dataset
* Preprocess features
* Train models
* Save trained models to `/models`

---

## 🧪 Run Tests

```bash
pytest -q
```

---

## 🔮 Future Improvements

* Add deep learning models (LSTM / CNN)
* Implement REST API with FastAPI
* Real-time traffic capture with Scapy
* Visualization dashboard

---
Computational resources
The requirements regarding the computational resources to train the classifiers are given below:

Category	Resource
CPU	Intel Core i7 processor
RAM	32 GB
GPU	1 GPU, 8 GB RAM
HDD	100 GB

## ⚡ References / Credits

* Original project inspiration: [cstub/ml-ids](https://github.com/cstub/ml-ids)
* This repo is a modified, extended version with additional features, preprocessing, and evaluation.



