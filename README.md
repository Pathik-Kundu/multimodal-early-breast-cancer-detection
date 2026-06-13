# Dual-Stream Framework for Early Breast Cancer Detection: Synergizing Classical Machine Learning with Deep Autoencoders

> Because why choose between clinical records and medical images when you can leverage both for high-precision diagnostic screening?

This repository contains the official implementation of the **Dual-Stream Framework**, a multimodal approach designed to advance early breast cancer detection. Developed by researchers at United International University (UIU), this framework treats diagnostic data as two distinct yet complementary streams: a structured clinical feature stream (analyzed via optimized classical machine learning) and an unstructured medical image stream (processed via deep unsupervised representation learning).

---

## 📌 Project Overview

Breast cancer remains a leading cause of global cancer-related mortality. Traditional diagnostic workflows are often prone to human variability, image quality fluctuations, and delayed timelines.

This project addresses these challenges by implementing a **Dual-Stream Architecture**:

1. **The Clinical Feature Stream:** Processes tabular clinical data (e.g., WDBC dataset) using a rigorous preprocessing pipeline—including multicollinearity reduction via Variance Inflation Factor (VIF), outlier removal via Isolation Forests, and minority class oversampling via SMOTE—culminating in an optimized Support Vector Machine (SVM) classifier.
2. **The Medical Image Stream:** Addresses data scarcity (lack of annotated malignant images) by using a Convolutional Autoencoder (CAE) for **Unsupervised Anomaly Detection**. The model learns the structural essence of healthy anatomy and flags anomalous (cancerous) tissues based on reconstruction error.

---

## 🛠️ Framework Architecture & Methodology

### 1. Clinical Feature Stream (Tabular Pipeline)

To ensure statistical robustness and prevent model instability from highly correlated features, the structured data undergoes a seven-stage pipeline:

* **Multicollinearity Reduction:** Features with a Variance Inflation Factor ($VIF$) greater than 10 are systematically removed to prevent variance inflation of regression coefficients:

$$VIF_{i} = \frac{1}{1-R_{i}^{2}}$$


* **Outlier Detection:** Handled via the **Isolation Forest** algorithm to eliminate measurement artifacts while preserving crucial biological anomalies.
* **Class Balancing (SMOTE):** Corrects class imbalance (~63% Benign vs. ~37% Malignant) on the training set by interpolating new minority samples:

$$x_{new} = x_{i} + \lambda \times (x_{neighbor} - x_{i})$$


* **Classification:** Implements an SVM minimizing the soft-margin objective function:

$$\min_{w,b,\zeta}\frac{1}{2}||w||^{2}+C\sum_{i=1}^{n}\zeta_{i}$$



Utilizes a Radial Basis Function (RBF) kernel, $K(x,x^{\prime})=\exp(-\gamma||x-x^{\prime}||^{2})$, to effectively handle non-linear decision boundaries.

### 2. Medical Image Stream (Deep Learning Pipeline)

* **Convolutional Autoencoder (CAE):** Trained *exclusively* on normal (benign) images. When presented with an anomaly (malignancy), the reconstruction fidelity drops.
* **Objective Function:** Minimizes Mean Squared Error (MSE) loss:

$$\mathcal{L}(X,\hat{X})=\frac{1}{N}\sum_{i=1}^{N}||X_{i}-\hat{X_{i}}||^{2}$$


* **Latent Space Exploration:** Bottleneck features are compressed and clustered using K-Means ($k=2$) and visualized via Principal Component Analysis (PCA) to evaluate unsupervised class separation.

---

## 📈 Performance & Key Results

### Tabular Stream Performance Comparison

The SVM with an RBF kernel proved to be the superior model, balancing near-perfect sensitivity with flawless specificity.

| Model | Accuracy | False Negatives (FN) | False Positives (FP) | AUC Score |
| --- | --- | --- | --- | --- |
| **SVM (RBF Kernel)** | **97.37%** | **3** | **0** | **0.99** |
| Logistic Regression | 96.49% | 3 | 1 | 1.00 |
| Random Forest | 96.49% | 4 | 0 | 0.99 |
| Gradient Boosting | 96.49% | 4 | 0 | 1.00 |
| KNN ($k=7$) | 95.61% | 4 | 1 | 0.98 |

> **Key Takeaway:** The SVM model achieved a perfect specificity (1.00), meaning **zero** healthy patients were misdiagnosed with false positives, effectively eliminating unnecessary diagnostic stress.

### Image Stream Performance

* **Reconstruction Fidelity:** Reached an exceptional **99% similarity score** on both training and testing datasets for benign tissue structure.
* **Dynamic Thresholding:** Successfully flagged structural variations as statistical anomalies using a dynamic threshold ($\mu_{train\_loss} + \sigma_{train\_loss}$).

---

## 📦 Repository Structure

```text
├── Image_Data_code.ipynb     # Convolutional Autoencoder for unsupervised image anomaly detection
├── Tabular_ Data_code.ipynb   # VIF, SMOTE, and classical ML classifiers (SVM, RF, etc.)
├── Tabular Dataset/
    └── breast-cancer.csv      # Structured diagnostic clinical dataset
├── Image Dataset Link Kaggle.txt   #Download image datset from the link
└── requirements.txt    #requirements.txt for run the codes

```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Clone this repository and install the mandatory requirements:

```bash
git clone https://github.com/Pathik-Kundu/multimodal-early-breast-cancer-detection.git
cd multimodal-early-breast-cancer-detection
pip install -r requirements.txt

```

### Quick Usage Example (Tabular Stream)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv('your_wdbc_dataset.csv')
data.drop("id", axis=1, inplace=True)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Best Model
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train_scaled, y_train)
print(f"Model Training Complete. Test Accuracy: {svm_model.score(X_test_scaled, y_test):.4f}")

```

---

## 🔮 Future Improvements

* **Multimodal Fusion:** Transitioning from isolated silos to an End-to-End network that simultaneously accepts mammography scans and clinical history to compute a unified risk score.
* **Explainable AI (XAI):** Implementing Grad-CAM to output interpretable visual heatmaps pinpointing exactly where the autoencoder detects tissue anomalies.
* **Edge Optimization:** Quantizing deep learning weights from float32 to int8 to allow real-time inference on hospital tablets in low-resource rural settings.

---

## 👥 Authors & Affiliation

**Shahriar Islam, Abrar Jaman Gazi, Pathik Kundu, Abu Nasir Patwary, and Manzil Ahsan**
*Department of Computer Science and Engineering, United International University (UIU), Dhaka, Bangladesh.*
