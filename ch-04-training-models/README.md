# Chapter 4 — Training Models

This chapter covers the fundamental concepts and algorithms used to train Machine Learning models. The implementations are based on **Chapter 4: Training Models** from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.

The chapter focuses on understanding how models learn from data, how different optimization algorithms work, and how regularization can be used to control overfitting.

---

## 📂 Chapter Structure

```text
ch-04-training-models/
│
├── models.ipynb
└── README.md
```

### Notebook

- [`models.ipynb`](./models.ipynb) — Implementations and experiments covering the main concepts from Chapter 4.

---

## 📚 Topics Covered

### 1. Linear Regression

Linear Regression is used to model the relationship between input features and a continuous target variable.

The notebook explores:

- Linear Regression
- The Normal Equation
- Making predictions
- Mean Squared Error
- Model parameters
- Linear Regression using Scikit-Learn

The basic linear model is:

```text
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

---

### 2. The Normal Equation

The Normal Equation provides a closed-form solution for Linear Regression.

```text
θ = (XᵀX)⁻¹Xᵀy
```

It allows the model parameters to be calculated directly without using an iterative optimization algorithm.

The notebook demonstrates how the Normal Equation can be used to obtain the optimal parameters for a linear model.

---

### 3. Computational Complexity

The computational cost of the Normal Equation increases significantly as the number of features grows.

This motivates the use of iterative optimization algorithms such as **Gradient Descent**, especially when dealing with datasets containing a large number of features.

---

## 4. Gradient Descent

Gradient Descent is an iterative optimization algorithm used to minimize a model's cost function.

The general update rule is:

```text
θ := θ − η∇θ MSE(θ)
```

where:

- `θ` — model parameters
- `η` — learning rate
- `∇θ MSE(θ)` — gradient of the cost function

The notebook explores the effect of the learning rate and the movement of the parameters toward the minimum of the cost function.

---

### Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the **entire training set** during every iteration.

```python
gradients = 2 / m * X_b.T @ (X_b @ theta - y)

theta = theta - eta * gradients
```

The learning rate controls the size of each update.

---

### Learning Rate

The learning rate is one of the most important Gradient Descent hyperparameters.

A learning rate that is:

- **Too small** → training can take a very long time.
- **Too large** → the algorithm may overshoot the minimum or diverge.
- **Appropriate** → the algorithm converges toward the minimum efficiently.

Feature scaling is particularly important because features with very different scales can produce an elongated cost function.

---

## 5. Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) updates the model parameters using individual training instances instead of the entire dataset.

Compared with Batch Gradient Descent:

| Batch Gradient Descent           | Stochastic Gradient Descent   |
| -------------------------------- | ----------------------------- |
| Uses the complete dataset        | Uses one instance at a time   |
| More stable updates              | More irregular updates        |
| Can be slower for large datasets | Often faster per update       |
| Moves smoothly toward minimum    | Can bounce around the minimum |

SGD can also escape local irregularities because of its noisy updates.

---

## 6. Mini-Batch Gradient Descent

Mini-Batch Gradient Descent lies between Batch Gradient Descent and Stochastic Gradient Descent.

Instead of using:

```text
Entire dataset
```

or:

```text
One training example
```

it uses a small batch of training examples.

```text
Dataset
   ↓
Mini-batch
   ↓
Calculate gradient
   ↓
Update parameters
```

Mini-batch methods are particularly useful when working with larger datasets and hardware that can efficiently process batches.

---

# 7. Polynomial Regression

Linear Regression can also be used to fit nonlinear relationships by adding polynomial features.

For example:

```text
y = 0.5x² + 3x + 5 + noise
```

Polynomial features can transform:

```text
x
```

into:

```text
x, x²
```

using:

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly_features.fit_transform(X)
```

A Linear Regression model can then be trained on these transformed features.

This allows a linear model to fit a nonlinear curve.

---

## 8. Learning Curves

Learning curves help analyze how model performance changes as the size of the training set increases.

They can help identify:

- Underfitting
- Overfitting
- High bias
- High variance

A model suffering from underfitting generally performs poorly on both the training and validation sets.

A model suffering from overfitting can perform very well on the training set while performing considerably worse on unseen data.

---

# 9. Regularized Linear Models

Regularization is used to reduce overfitting by constraining the model parameters.

The chapter explores:

- Ridge Regression
- Lasso Regression
- Elastic Net

Regularization adds a penalty to the cost function.

---

## 10. Ridge Regression

Ridge Regression uses **L2 regularization**.

Its cost function adds a penalty based on the squared magnitude of the model coefficients.

```text
Cost = MSE + α Σθᵢ²
```

In Scikit-Learn:

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)
```

The parameter `alpha` controls the strength of regularization.

Higher `alpha` generally results in stronger regularization and smaller model coefficients.

---

## 11. Lasso Regression

Lasso Regression uses **L1 regularization**.

Its penalty is based on the absolute values of the model coefficients.

```text
Cost = MSE + α Σ|θᵢ|
```

In Scikit-Learn:

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
```

One important property of Lasso is that it can drive some coefficients exactly to zero.

This can make Lasso useful for feature selection.

---

## 12. Elastic Net

Elastic Net combines both L1 and L2 regularization.

```text
Elastic Net
     │
     ├── L1 penalty
     │
     └── L2 penalty
```

It provides a balance between the behavior of Lasso and Ridge Regression.

In Scikit-Learn:

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic_net.fit(X, y)
```

The `l1_ratio` determines the balance between L1 and L2 regularization.

---

# 13. Model Comparison

The major models studied in this chapter can be summarized as follows:

| Model                 | Main Idea                           | Regularization |
| --------------------- | ----------------------------------- | -------------- |
| Linear Regression     | Fits a linear relationship          | None           |
| Polynomial Regression | Adds polynomial features            | None           |
| Ridge Regression      | Penalizes large coefficients        | L2             |
| Lasso Regression      | Encourages coefficients toward zero | L1             |
| Elastic Net           | Combines Ridge and Lasso            | L1 + L2        |

---

# 14. Important Concepts

### Cost Function

A cost function measures how well a model's predictions match the actual target values.

For Linear Regression, Mean Squared Error is commonly used:

```text
MSE = (1/m) Σ(ŷᵢ − yᵢ)²
```

### Learning Rate

Controls the size of parameter updates during Gradient Descent.

### Convergence

The optimization algorithm is considered to have converged when the parameters approach a point where further updates produce little improvement.

### Overfitting

A model performs very well on the training data but poorly on unseen data.

### Underfitting

A model is too simple to capture the underlying relationships in the data.

### Regularization

A technique used to constrain model complexity and reduce overfitting.

---

# 🧪 Experiments in This Chapter

The notebook contains practical experiments involving:

- Linear Regression
- The Normal Equation
- Gradient Descent
- Learning-rate behavior
- Polynomial Regression
- Polynomial feature generation
- Learning curves
- Ridge Regression
- Lasso Regression
- Regularization and model complexity

The experiments are implemented using Python and Scikit-Learn along with NumPy and Matplotlib.

---

# 🛠️ Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-Learn**
- **Jupyter Notebook**

---

# 🎯 Learning Objectives

After completing this chapter, the main concepts to understand are:

- How Linear Regression works internally
- How the Normal Equation finds model parameters
- Why Gradient Descent is useful
- How the learning rate affects optimization
- Difference between Batch, Stochastic, and Mini-Batch Gradient Descent
- How polynomial features allow Linear Regression to model nonlinear relationships
- How learning curves help diagnose model performance
- Why regularization is necessary
- Difference between Ridge and Lasso Regression
- How Elastic Net combines L1 and L2 regularization

---

## 📖 Reference

This chapter follows the concepts presented in:

**Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow**

by Aurélien Géron.

---

## 🔗 Repository

This chapter is part of my ongoing implementation and learning repository:

[hands_on_ml — GitHub Repository](https://github.com/ombhiwaniya911/hands_on_ml?utm_source=chatgpt.com)

