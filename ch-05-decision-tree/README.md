# Chapter 5 — Decision Trees

This chapter explores **Decision Tree algorithms** using Scikit-Learn. Decision Trees are powerful supervised learning algorithms that can be used for both **classification and regression**. They are also important building blocks for ensemble methods such as Random Forests.

This chapter is part of my implementation and learning journey based on **_Hands-On Machine Learning with Scikit-Learn & PyTorch by Aurélien Géron**.

---

## 📚 Topics Covered

- Decision Tree Classification
- Training a `DecisionTreeClassifier`
- Visualizing Decision Trees
- Making predictions
- Estimating class probabilities
- Gini impurity
- Entropy
- CART (Classification and Regression Trees)
- Computational complexity
- Decision Tree regularization
- Hyperparameters
- Decision Tree Regression
- `DecisionTreeRegressor`
- Overfitting and underfitting
- Decision Tree instability
- Decision boundaries
- Understanding tree-based models

---

## 🌳 What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that makes predictions by repeatedly splitting the dataset based on feature values.

A simple tree can be thought of as a sequence of questions:

```text
Is petal width <= 0.8?
       /       \
     Yes        No
     /           \
 Class 0       Is petal width <= 1.75?
                  /       \
                Yes        No
                /           \
             Class 1      Class 2
```

Each internal node represents a decision, while each leaf node represents the final prediction.

Decision Trees can naturally model nonlinear relationships without requiring feature transformations such as polynomial features.

---

## 🧪 Classification

The chapter begins with classification using the **Iris dataset**.

Example:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

tree_clf = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

tree_clf.fit(X, y)
```

The trained model can then be used for predictions:

```python
tree_clf.predict([[5, 1.5]])
```

---

## 📊 Visualizing a Decision Tree

Decision Trees have the advantage of being relatively easy to interpret and visualize.

Scikit-Learn provides tools such as:

```python
from sklearn.tree import export_graphviz
```

A `.dot` file can be generated and rendered using **Graphviz**.

The visualization makes it possible to see:

- The feature used for splitting
- The splitting threshold
- Number of samples
- Class distribution
- Gini impurity
- Predicted class

---

## 🎯 Gini Impurity

For classification, Scikit-Learn uses **Gini impurity** by default.

Gini impurity measures how mixed the classes are within a node.

For a node containing several classes:

\[
G_i = 1 - \sum_{k=1}^{n} p_k^2
\]

where:

- \(p_k\) is the proportion of samples belonging to class \(k\)
- \(n\) is the number of classes

A node containing samples from only one class has:

\[
G_i = 0
\]

which means the node is completely pure.

---

## 🔀 Entropy

Another criterion that can be used for measuring impurity is **entropy**.

The entropy of a node is:

\[
H = -\sum_{k=1}^{n}p_k\log_2(p_k)
\]

A Decision Tree can be configured to use entropy instead of Gini impurity:

```python
tree_clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    random_state=42
)
```

Both Gini impurity and entropy can be used to determine good splits.

---

## ⚙️ CART Algorithm

Scikit-Learn uses an optimized version of the **Classification And Regression Tree (CART)** algorithm.

The algorithm recursively searches for feature/threshold combinations that produce the best split according to the selected impurity or error criterion.

The general process is:

```text
Dataset
   ↓
Find best feature + threshold
   ↓
Split dataset
   ↓
Repeat for each child node
   ↓
Stop according to constraints
   ↓
Create leaf nodes
```

---

## 🛡️ Regularization

Decision Trees can easily **overfit** the training data because they can continue splitting until they create extremely specific rules.

For example:

```python
DecisionTreeClassifier()
```

may produce a very deep tree.

Regularization parameters can be used to restrict the complexity of the tree:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_leaf_nodes=10,
    random_state=42
)
```

Important hyperparameters include:

| Parameter | Purpose |
|---|---|
| `max_depth` | Limits the maximum depth of the tree |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required in a leaf |
| `max_leaf_nodes` | Limits the number of leaf nodes |
| `criterion` | Determines how splits are evaluated |

---

## 📈 Decision Tree Regression

Decision Trees can also be used for regression.

Scikit-Learn provides:

```python
from sklearn.tree import DecisionTreeRegressor
```

Example:

```python
tree_reg = DecisionTreeRegressor(
    max_depth=2,
    random_state=42
)

tree_reg.fit(X, y)
```

Instead of predicting a class, each leaf predicts a numerical value.

For regression, the tree attempts to create splits that minimize the prediction error within the resulting regions.

---

## 🌙 Decision Trees and Nonlinear Data

Decision Trees are particularly useful for datasets where the relationship between features and target values is nonlinear.

For example, the `make_moons` dataset can be used to demonstrate nonlinear decision boundaries:

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=500,
    noise=0.30,
    random_state=42
)
```

A Decision Tree can learn nonlinear boundaries without explicitly creating polynomial features.

---

## ⚠️ Decision Tree Limitations

Although Decision Trees are powerful, they have some important limitations.

### 1. Overfitting

Deep trees can memorize the training data.

### 2. High Variance

Small changes in the training data can result in a significantly different tree.

### 3. Axis-Aligned Boundaries

Standard Decision Trees make splits based on one feature at a time, which can result in axis-aligned decision boundaries.

### 4. Instability

Because of their sensitivity to the training data, individual Decision Trees can be unstable.

These limitations are some of the reasons why ensemble methods such as **Random Forests** are useful.

---

## 🛠️ Libraries Used

The notebooks use Python and common machine-learning libraries:

```text
Python
NumPy
Pandas
Matplotlib
Scikit-Learn
Graphviz
```

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn graphviz
```

> Graphviz itself may also need to be installed separately on your operating system.

---

## 📂 Chapter Structure

```text
ch-05-decision-tree/
│
├── decision_tree.ipynb
└── README.md
```

The notebook contains the practical implementation and experiments for the concepts discussed in this chapter.

---

## 🎯 Learning Objectives

After completing this chapter, I aim to be able to:

- Understand how Decision Trees work internally
- Train classification and regression trees
- Visualize trained Decision Trees
- Interpret tree splits and leaf nodes
- Understand Gini impurity and entropy
- Understand the CART algorithm
- Control overfitting using regularization
- Tune Decision Tree hyperparameters
- Create nonlinear decision boundaries
- Understand the strengths and limitations of Decision Trees

---

## 📖 Reference

This chapter is based on:

**Aurélien Géron — _Hands-On Machine Learning with Scikit-Learn & PyTorch**

The official book's Decision Tree chapter covers training and visualization, predictions, class probabilities, CART, impurity criteria, regularization, regression, and tree instability.

---

## 🔗 Repository

Main repository:

[hands_on_ml — GitHub](https://github.com/ombhiwaniya911/hands_on_ml?utm_source=chatgpt.com)

Chapter:

[Chapter 5 — Decision Tree](https://github.com/ombhiwaniya911/hands_on_ml/tree/main/ch-05-decision-tree?utm_source=chatgpt.com)

---

## 🚀 Next Chapter

The next chapter focuses on **Ensemble Learning and Random Forests**, building on Decision Trees to create stronger and more robust models.
