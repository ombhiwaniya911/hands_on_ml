# Chapter 6 — Ensemble Learning and Random Forests

This chapter explores **Ensemble Learning and Random Forests** using Scikit-Learn. Ensemble methods combine the predictions of multiple machine learning models to create a stronger and often more robust predictor.

This chapter is part of my implementation and learning journey based on **_Hands-On Machine Learning with Scikit-Learn and PyTorch_ by Aurélien Géron**. The book's Chapter 6 covers voting classifiers, bagging and pasting, random forests, Extra-Trees, feature importance, boosting, and stacking.

---

## 📚 Topics Covered

- Ensemble Learning
- Voting Classifiers
  - Hard Voting
  - Soft Voting
- Bagging and Pasting
- Bagging in Scikit-Learn
- Out-of-Bag Evaluation
- Random Patches
- Random Subspaces
- Random Forests
- Extra-Trees
- Feature Importance
- Boosting
  - AdaBoost
  - Gradient Boosting
  - Histogram-Based Gradient Boosting
- Stacking
- Comparing Ensemble Methods

---

## 🤝 What is Ensemble Learning?

**Ensemble Learning** is a technique where multiple predictors are combined to produce a better final prediction.

Instead of relying on a single model:

```text
          Training Data
               │
               ▼
        ┌──────────────┐
        │ Single Model │
        └──────┬───────┘
               │
               ▼
          Prediction
```

an ensemble combines multiple models:

```text
                  Training Data
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Model 1      Model 2      Model 3
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Combination
                       │
                       ▼
                 Final Prediction
```

The idea is that different models may make different errors. Combining them can therefore produce a more accurate and robust model.

---

# 🗳️ Voting Classifiers

A voting classifier combines several different classifiers and uses their predictions to make the final decision.

For example:

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(random_state=42))
    ]
)
```

### Hard Voting

In **hard voting**, each classifier votes for a class and the class receiving the most votes becomes the final prediction.

```text
Classifier 1 → Class 1
Classifier 2 → Class 1
Classifier 3 → Class 0

Final Prediction → Class 1
```

### Soft Voting

Soft voting uses the predicted class probabilities instead of simply counting votes.

```text
Classifier 1 → [0.20, 0.80]
Classifier 2 → [0.30, 0.70]
Classifier 3 → [0.10, 0.90]

Average     → [0.20, 0.80]

Final       → Class 1
```

Soft voting generally works best when the individual classifiers can provide reliable probability estimates.

---

# 🎒 Bagging and Pasting

**Bagging** and **pasting** train multiple models using different subsets of the training data.

### Bagging

Bagging stands for:

> **Bootstrap Aggregating**

Each predictor is trained on a randomly selected subset of the training data **with replacement**.

```text
Original Dataset
       │
       ├── Bootstrap Sample 1 → Model 1
       ├── Bootstrap Sample 2 → Model 2
       ├── Bootstrap Sample 3 → Model 3
       └── Bootstrap Sample 4 → Model 4
                         │
                         ▼
                    Aggregation
                         │
                         ▼
                  Final Prediction
```

### Pasting

Pasting is similar to bagging, except the samples are selected **without replacement**.

| Method | Sampling |
|---|---|
| Bagging | With replacement |
| Pasting | Without replacement |

---

# 🧪 Bagging with Scikit-Learn

Scikit-Learn provides `BaggingClassifier` for creating bagging ensembles.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    random_state=42
)

bag_clf.fit(X_train, y_train)
```

The `n_estimators` parameter controls the number of individual predictors in the ensemble.

---

# 🔍 Out-of-Bag Evaluation

When using bagging, each bootstrap sample contains duplicate observations because sampling is performed with replacement.

As a result, some training instances are not selected for a particular predictor.

These unused samples are called **Out-of-Bag (OOB)** instances.

They can be used to evaluate the model without requiring a separate validation set.

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    oob_score=True,
    random_state=42
)

bag_clf.fit(X_train, y_train)

bag_clf.oob_score_
```

OOB evaluation provides a convenient estimate of the model's generalization performance.

---

# 🌲 Random Forests

A **Random Forest** is an ensemble of Decision Trees.

Instead of training a single Decision Tree, Random Forest trains many trees using randomized subsets of the data and features.

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    random_state=42
)

rnd_clf.fit(X_train, y_train)
```

The final prediction is obtained by aggregating the predictions of all the individual trees.

```text
                Dataset
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     Tree 1     Tree 2     Tree 3
        │          │          │
        └──────────┼──────────┘
                   ▼
              Aggregation
                   │
                   ▼
            Final Prediction
```

Random Forests are among the most useful traditional machine learning algorithms because they can model nonlinear relationships and usually require relatively little preprocessing.

---

# 🌳 Extra-Trees

**Extra-Trees**, or Extremely Randomized Trees, introduce additional randomness into the tree-building process.

They are implemented using:

```python
from sklearn.ensemble import ExtraTreesClassifier

extra_trees_clf = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42
)
```

Extra-Trees can sometimes provide better performance than Random Forests while also being computationally efficient.

---

# ⭐ Feature Importance

Random Forests can be used to estimate the importance of individual features.

```python
rnd_clf.feature_importances_
```

For example:

```python
for name, score in zip(feature_names, rnd_clf.feature_importances_):
    print(name, score)
```

This can help identify which features contribute most to the model's predictions.

Feature importance is particularly useful for:

- Feature selection
- Understanding datasets
- Model interpretation
- Removing irrelevant features

---

# 🚀 Boosting

**Boosting** is another ensemble technique.

Instead of training many independent models and combining them, boosting trains models **sequentially**, with each new model attempting to improve upon the previous models.

```text
Model 1
   │
   ▼
Errors
   │
   ▼
Model 2
   │
   ▼
Errors
   │
   ▼
Model 3
   │
   ▼
Final Ensemble
```

The chapter covers several boosting techniques.

---

# ⚡ AdaBoost

**AdaBoost** stands for Adaptive Boosting.

It focuses more attention on the training instances that previous predictors classified incorrectly.

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    n_estimators=30,
    random_state=42
)

ada_clf.fit(X_train, y_train)
```

The models are trained sequentially, and their contributions are weighted according to their performance.

---

# 📈 Gradient Boosting

Gradient Boosting also trains predictors sequentially.

Each new predictor attempts to reduce the errors made by the previous ensemble.

A simple implementation is:

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbrt.fit(X_train, y_train)
```

Important hyperparameters include:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `max_leaf_nodes`

---

# ⚡ Histogram-Based Gradient Boosting

Scikit-Learn also provides **Histogram-Based Gradient Boosting**.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hist_clf = HistGradientBoostingClassifier(
    random_state=42
)

hist_clf.fit(X_train, y_train)
```

Histogram-based methods can be significantly faster on large datasets because continuous features are divided into discrete bins.

---

# 🧩 Stacking

**Stacking**, or stacked generalization, combines multiple different predictors using another model called a **final estimator**.

```text
                 Input Data
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Model 1        Model 2      Model 3
        │            │            │
        └────────────┼────────────┘
                     ▼
              Predictions
                     │
                     ▼
              Final Estimator
                     │
                     ▼
              Final Prediction
```

Scikit-Learn provides:

```python
from sklearn.ensemble import StackingClassifier
```

Example:

```python
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression()),
        ("rf", RandomForestClassifier()),
        ("svc", SVC())
    ],
    final_estimator=LogisticRegression()
)

stacking_clf.fit(X_train, y_train)
```

The final estimator learns how to combine the predictions of the base models.

---

# 🧠 Ensemble Methods Compared

| Method | Main Idea |
|---|---|
| Voting | Combine predictions from different models |
| Bagging | Train models on bootstrap samples |
| Pasting | Train models on random samples without replacement |
| Random Forest | Ensemble of randomized Decision Trees |
| Extra-Trees | Highly randomized Decision Trees |
| AdaBoost | Sequentially focus on difficult examples |
| Gradient Boosting | Sequentially minimize prediction errors |
| Stacking | Use another model to combine predictions |

---

# 🛠️ Libraries Used

The experiments in this chapter primarily use:

```text
Python
NumPy
Pandas
Matplotlib
Scikit-Learn
```

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# 📂 Chapter Structure

```text
ch-06-ensemble-models/
│
├── ensemble.ipynb
└── README.md
```

The notebook contains the practical implementations and experiments related to the concepts covered in this chapter.

---

# 🎯 Learning Objectives

After completing this chapter, I aim to be able to:

- Understand the idea behind Ensemble Learning
- Build voting classifiers
- Understand hard and soft voting
- Implement Bagging and Pasting
- Perform Out-of-Bag evaluation
- Understand Random Forests
- Use Extra-Trees
- Calculate feature importance
- Understand AdaBoost
- Understand Gradient Boosting
- Work with Histogram-Based Gradient Boosting
- Understand Stacking
- Compare different ensemble learning techniques
- Understand why combining models can improve generalization

---

# 📖 Reference

This chapter follows:

**Aurélien Géron — _Hands-On Machine Learning with Scikit-Learn and PyTorch_**

Chapter 6 is titled **“Ensemble Learning and Random Forests”** and covers voting classifiers, bagging and pasting, Random Forests, Extra-Trees, feature importance, boosting, and stacking.

---

# 🔗 Repository

Main repository:

[Hands-On ML — GitHub](https://github.com/ombhiwaniya911/hands_on_ml)

Chapter 6:

[Chapter 6 — Ensemble Models](https://github.com/ombhiwaniya911/hands_on_ml/tree/main/ch-06-ensemble-models)

---

## 🚀 Next Chapter

The next chapter focuses on **Dimensionality Reduction**, exploring techniques for reducing the number of features while retaining as much useful information as possible.
