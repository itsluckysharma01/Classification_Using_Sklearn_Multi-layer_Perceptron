# 🧠 Classification Using Sklearn Multi-layer Perceptron

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/itsluckysharma01/Classification_Using_Sklearn_Multi-layer_Perceptron)

---

## 📌 Overview

This project demonstrates **breast cancer classification** using a **Multi-Layer Perceptron (MLP)** neural network implemented with [scikit-learn](https://scikit-learn.org/). The model is trained on the well-known **Wisconsin Breast Cancer Dataset** to predict whether a tumour is **malignant** or **benign**.

Multi-Layer Perceptrons are a type of feedforward neural network capable of learning non-linear decision boundaries, making them well-suited for complex classification problems like medical diagnosis.

---

## 🚀 Key Features

- ✅ Loads the built-in breast cancer dataset from scikit-learn
- ✅ Splits data into training and test sets
- ✅ Applies `StandardScaler` for feature normalisation
- ✅ Trains an `MLPClassifier` with two hidden layers (64 → 32 neurons)
- ✅ Evaluates accuracy, precision, recall, and F1-score
- ✅ Saves the trained model using `joblib` for later reuse

---

## 🏗️ How MLP Classification Works

```
Input Layer  →  Hidden Layer 1 (64 neurons)  →  Hidden Layer 2 (32 neurons)  →  Output Layer
```

| Step | Description |
|------|-------------|
| **Initialisation** | Weights and biases are seeded with small random values |
| **Forward Propagation** | Data flows through layers; activation functions introduce non-linearity |
| **Loss Calculation** | Cross-entropy loss measures prediction quality |
| **Back Propagation** | Gradient descent updates weights to minimise the loss |
| **Prediction** | Trained network predicts class labels on unseen data |

---

## 📁 Project Structure

```
Classification_Using_Sklearn_Multi-layer_Perceptron/
│
├── Classification_Using_Sklearn_Multi-layer_Perceptron.ipynb   # Main notebook
├── breast_cancer_mlp_model.pkl                                  # Saved model
├── requirements.txt                                             # Python dependencies
└── README.md                                                    # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/itsluckysharma01/Classification_Using_Sklearn_Multi-layer_Perceptron.git
cd Classification_Using_Sklearn_Multi-layer_Perceptron
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook Classification_Using_Sklearn_Multi-layer_Perceptron.ipynb
```

---

## 🧪 Usage

Run all cells in the notebook sequentially, or copy the snippet below to get started:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# 2. Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# 4. Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)

# 5. Evaluate
y_pred = mlp.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.4f}%")
print(classification_report(y_test, y_pred))
```

### 💾 Save the Trained Model

```python
import joblib
joblib.dump(mlp, 'breast_cancer_mlp_model.pkl')
```

### 🔄 Load and Reuse the Saved Model

```python
import joblib
model = joblib.load('breast_cancer_mlp_model.pkl')
predictions = model.predict(x_test)
```

---

## 📊 Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 2 |
| Neurons — Layer 1 | 64 |
| Neurons — Layer 2 | 32 |
| Max Iterations | 1000 |
| Random State | 42 |
| Solver | `adam` (default) |
| Activation Function | `relu` (default) |

---

## 📈 Expected Results

> Results may vary slightly depending on the scikit-learn version.

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97–98% |
| **Precision** (Malignant) | High |
| **Recall** (Malignant) | High |
| **F1-Score** | High |

The model achieves high precision and recall for both **malignant** and **benign** classes, demonstrating strong performance on this medical classification task.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | Wisconsin Breast Cancer (Diagnostic) |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| **Access** | `sklearn.datasets.load_breast_cancer()` |
| **Samples** | 569 |
| **Features** | 30 (mean, SE, and worst of 10 cell nucleus measurements) |
| **Classes** | 2 — Malignant (0) and Benign (1) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙋 Author

**Lucky Sharma**  
GitHub: [@itsluckysharma01](https://github.com/itsluckysharma01)

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
