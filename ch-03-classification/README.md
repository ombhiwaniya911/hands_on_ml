# Chapter 3 — Classification

This chapter focuses on **classification algorithms and evaluation techniques** using the **MNIST handwritten digit dataset**.

The notebook follows the classification concepts from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* and implements the concepts using Python and Scikit-Learn.

## 📓 Notebook

- [`Untitled.ipynb`](./Untitled.ipynb) — Complete implementation and experiments

## 📊 Dataset

The notebook uses the **MNIST dataset**, loaded using Scikit-Learn's `fetch_openml()`.

The dataset contains:

- **70,000** handwritten digit images
- Each image is **28 × 28 pixels**
- **784 features** per image
- Digits from **0 to 9**

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist.data, mnist.target
```

## 🎯 Topics Covered

### 1. Exploring the MNIST Dataset

- Loading the MNIST dataset
- Understanding the feature matrix and target labels
- Visualizing handwritten digits
- Reshaping the 784-pixel feature vector into a 28 × 28 image

### 2. Binary Classification

The notebook converts the multiclass MNIST problem into a binary classification problem:

> **Is the digit a 5 or not?**

```python
y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5")
```

This introduces the difference between **binary** and **multiclass classification**.

### 3. SGD Classifier

An `SGDClassifier` is trained to recognize the digit 5.

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=38)
sgd_clf.fit(x_train, y_train_5)
```

Concepts explored include:

- Stochastic Gradient Descent
- Binary classification
- Decision scores
- Predictions
- Model training iterations

### 4. Performance Evaluation

Different classification metrics and evaluation techniques are explored, including:

- Cross-validation
- Accuracy
- Confusion matrix
- Precision
- Recall
- F1 score

The confusion matrix is used to understand:

- True Positives
- True Negatives
- False Positives
- False Negatives

### 5. Decision Scores and Thresholds

The notebook explores how a classifier's **decision function** produces scores and how changing the classification threshold affects predictions.

This demonstrates the trade-off between:

- Precision
- Recall

and how the threshold can be selected depending on the desired performance.

### 6. Precision-Recall Trade-off

The notebook examines how increasing or decreasing the decision threshold changes precision and recall.

This is useful when accuracy alone is not sufficient to evaluate a classifier.

### 7. ROC Curve

The **Receiver Operating Characteristic (ROC) curve** is implemented using:

- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Classification thresholds

The notebook also examines a threshold corresponding to a target precision level.

### 8. Random Forest Classifier

A `RandomForestClassifier` is introduced as another classification algorithm.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
```

The Random Forest model is then compared using classification evaluation techniques.

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Matplotlib
- Scikit-Learn
- OpenML / MNIST

## 📁 Project Structure

```text
ch-03-classification/
│
├── Untitled.ipynb
├── .gitignore
└── README.md
```

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/ombhiwaniya911/hands_on_ml.git
```

Navigate to the chapter:

```bash
cd hands_on_ml/ch-03-classification
```

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Untitled.ipynb
```

## 📚 Key Concepts Learned

By completing this chapter, the following concepts are practiced:

- Classification vs. regression
- Binary classification
- Multiclass classification
- SGD-based classification
- Decision functions
- Classification thresholds
- Cross-validation
- Confusion matrices
- Precision
- Recall
- F1 score
- Precision/recall trade-off
- ROC curves
- False Positive Rate
- True Positive Rate
- Random Forest classification
- Model evaluation

## 🔗 Repository

**Hands-On ML Learning Repository:**  
https://github.com/ombhiwaniya911/hands_on_ml

**Chapter 3 Notebook:**  
https://github.com/ombhiwaniya911/hands_on_ml/blob/main/ch-03-classification/Untitled.ipynb

---

### Status

🚧 **In Progress** — This chapter is part of my ongoing implementation and learning journey through *Hands-On Machine Learning*.
