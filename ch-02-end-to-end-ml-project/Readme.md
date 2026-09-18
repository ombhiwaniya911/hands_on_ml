# California Housing Price Prediction

An end-to-end Machine Learning project for predicting **California housing prices** using Python and Scikit-Learn.

This project is based on the end-to-end Machine Learning workflow covered in *Hands-On Machine Learning with Scikit-Learn and PyTorch* by Aurélien Géron. It focuses on understanding the complete ML pipeline — from exploring raw data to preparing features, training models, and evaluating their performance.

## 📌 Project Overview

The goal of this project is to build a Machine Learning model that predicts the **median house value** of California districts using demographic, geographical, and housing-related features.

The project covers the major stages of a real-world Machine Learning workflow:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Train/Test Split
     ↓
Data Visualization
     ↓
Feature Analysis
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Selection & Tuning
```

## 📂 Project Structure

```text
ch-02-end-to-end-ml-project/
│
├── HousingPrice.ipynb
└── README.md
```

## 📊 Dataset

The project uses the **California Housing dataset**.

The dataset contains information about California districts, including:

* Longitude
* Latitude
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Median income
* Ocean proximity
* Median house value — target variable

### Target Variable

```text
median_house_value
```

The objective is to predict this value using the remaining features.

## 🔍 Data Exploration

The notebook explores the dataset using:

* Pandas
* NumPy
* Matplotlib
* Correlation analysis
* Histograms
* Geographical visualization
* Scatter plots

The analysis is used to understand:

* Feature distributions
* Missing values
* Outliers
* Relationships between features
* Correlation with the target variable

## 🛠️ Data Preprocessing

The project demonstrates several important preprocessing techniques.

### Handling Missing Values

Missing numerical values are handled using a median-based strategy.

### Categorical Features

The `ocean_proximity` categorical feature is converted into numerical features using:

```python
OneHotEncoder()
```

The project also demonstrates dropping the first category to avoid redundant dummy variables:

```python
OneHotEncoder(drop="first")
```

### Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

### Feature Engineering

Additional features are created to provide more useful information to the model:

```python
rooms_per_house
bedrooms_ratio
people_per_house
```

For example:

```python
rooms_per_house = total_rooms / households
```

These engineered features can provide more meaningful information than the original raw values alone.

## 🔄 Transformation Pipeline

Scikit-Learn pipelines and `ColumnTransformer` are used to organize preprocessing.

The pipeline handles:

```text
Numerical Features
       ↓
Imputation
       ↓
Feature Scaling
       ↓
Categorical Features
       ↓
One-Hot Encoding
       ↓
Prepared Dataset
```

This approach ensures that the same preprocessing steps can be consistently applied to training and test data.

## 🤖 Machine Learning Model

The project starts with a **Linear Regression** model.

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

The model learns the relationship between the processed housing features and the target variable:

```text
Housing Features → Linear Regression → Predicted House Value
```

Model predictions are evaluated against the actual house values.

## 📏 Model Evaluation

The project uses **Root Mean Squared Error (RMSE)** to evaluate regression performance.

```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(
    housing_labels,
    housing_predictions,
    squared=False
)
```

RMSE measures the typical magnitude of prediction errors, with larger errors receiving greater weight.

The project also explores model evaluation using **cross-validation**.

## 🚀 Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

## 📚 Concepts Covered

This project provides practical experience with:

* End-to-end Machine Learning workflow
* Exploratory Data Analysis
* Train/Test splitting
* Stratified sampling
* Data visualization
* Correlation analysis
* Feature engineering
* Missing-value imputation
* One-hot encoding
* Feature scaling
* `Pipeline`
* `ColumnTransformer`
* Linear Regression
* RMSE
* Cross-validation
* Model evaluation
* Hyperparameter tuning

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ombhiwaniya911/hands_on_ml.git
```

### 2. Navigate to the project

```bash
cd hands_on_ml/ch-02-end-to-end-ml-project
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Start Jupyter

```bash
jupyter notebook
```

### 5. Open

```text
HousingPrice.ipynb
```

and run the notebook cells sequentially.

## 📈 Future Improvements

Possible extensions to this project include:

* Compare Linear Regression with Decision Tree and Random Forest models
* Perform systematic hyperparameter tuning
* Improve feature engineering
* Analyze model errors
* Evaluate the final model on the untouched test set
* Save the trained model using `joblib`
* Build a prediction API
* Create a simple web interface for housing price predictions

## 📖 Reference

This project is based on concepts and workflow from:

**Aurélien Géron — Hands-On Machine Learning with Scikit-Learn and PyTorch**

The chapter follows the typical end-to-end process of understanding the problem, obtaining and exploring data, preparing the data, selecting/training models, evaluating them, and tuning the final solution.



⭐ This repository is part of my journey through *Hands-On Machine Learning with Scikit-Learn and PyTorch*, with each chapter implemented and documented as I learn.
