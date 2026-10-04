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

**Bagging** and **p
