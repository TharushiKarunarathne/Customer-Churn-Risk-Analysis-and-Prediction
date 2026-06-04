# Customer Churn Risk Analysis and Prediction

## Project Overview

This project analyzes customer churn data and builds machine learning models to predict whether a customer is likely to leave a company.

Customer churn is an important business problem because losing existing customers can reduce revenue and increase the cost of acquiring new customers. By identifying customers who are more likely to churn, a company can take early action and improve customer retention.

The main goal of this project is not only to build a prediction model, but also to understand the customer-related factors that may influence churn.

---

## Problem Statement

A company wants to understand which customers are more likely to stop using its service. Using customer information such as age, total purchase amount, account manager status, years with the company, and number of sites, this project builds classification models to predict customer churn risk.

The model can help the business:

- Identify high-risk customers early
- Improve customer retention strategies
- Support better decision-making for account managers
- Understand important churn-related factors

---

## Dataset

The dataset used in this project is a customer churn dataset available on Kaggle.

Dataset source: [Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/hassanamin/customer-churn)

The project uses two CSV files:

1. `customer_churn.csv`  
   This file contains historical customer data with the churn label.

2. `new_customers_1.csv`  
   This file contains new customer records without churn labels. The trained model is used to predict churn risk for these customers.

### Dataset Columns

| Column | Description |
|---|---|
| `Names` | Customer name |
| `Age` | Age of the customer |
| `Total_Purchase` | Total purchase amount made by the customer |
| `Account_Manager` | Whether the customer has an account manager |
| `Years` | Number of years the customer has been with the company |
| `Num_Sites` | Number of sites used by the customer |
| `Onboard_date` | Date the customer joined |
| `Location` | Customer location |
| `Company` | Customer company name |
| `Churn` | Target variable showing whether the customer churned |

Target column:

| Value | Meaning |
|---|---|
| `0` | Customer did not churn |
| `1` | Customer churned |

---

## Project Objectives

The main objectives of this project are:

- Understand the structure of the customer churn dataset
- Explore patterns between customer features and churn
- Clean and prepare the dataset for machine learning
- Train classification models to predict churn
- Compare model performance
- Identify important features that influence churn
- Predict churn risk for new customers
- Present final insights in a business-friendly way

---

## Project Workflow

The project follows these steps:

1. Data loading
2. Dataset inspection
3. Target variable analysis
4. Exploratory data analysis
5. Data cleaning and preprocessing
6. Feature selection
7. Train-test split
8. Logistic Regression model training
9. Random Forest model training
10. Model comparison
11. Model evaluation using classification metrics
12. ROC-AUC evaluation
13. Feature importance analysis
14. Model saving
15. New customer churn prediction
16. Final business insights

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand churn patterns and customer behavior.

The analysis focused on questions such as:

- Are older customers more likely to churn?
- Does total purchase amount affect churn?
- Are customers with more years less likely to churn?
- Does the number of sites influence churn?
- Does having an account manager reduce churn?

The following visualizations were created:

- Churn distribution
- Age vs churn
- Total purchase vs churn
- Years with company vs churn
- Number of sites vs churn
- Account manager vs churn

---

## Data Preprocessing

Before training the models, the dataset was cleaned and prepared.

The following columns were removed from the first version of the model:

- `Names`
- `Location`
- `Company`
- `Onboard_date`

These columns were removed because they are mainly text-based identifiers or date fields. This version of the project focuses on numerical customer behavior features.

The remaining features were used to train the classification models.

---

## Machine Learning Models

Two machine learning models were trained and compared.

### 1. Logistic Regression

Logistic Regression was used as the baseline model. It is simple, interpretable, and commonly used for binary classification problems.

### 2. Random Forest Classifier

Random Forest was used as a more advanced model. It can capture non-linear patterns and also provides feature importance values, which help explain which features are more useful for prediction.

---

## Model Evaluation

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC score
- ROC curve

For churn prediction, recall is especially important because the business wants to correctly identify customers who are likely to leave.

---

## Feature Importance

Feature importance was analyzed using the Random Forest model.

This helped identify which customer features contributed most to churn prediction. These insights can help the business understand customer behavior and improve retention strategies.

Important features may include:

- Number of sites
- Years with the company
- Total purchase amount
- Age
- Account manager status

---

## New Customer Churn Prediction

After training the final model, it was used to predict churn risk for new customers.

The model generated:

- Predicted churn class
- Churn probability

The prediction results were saved into a new CSV file:

`new_customer_churn_predictions.csv`

This output can help a business identify customers who may need extra support or attention.

---

## Project Structure

Customer-Churn-Risk-Analysis/

- data/
  - customer_churn.csv
  - new_customers_1.csv
  - new_customer_churn_predictions.csv

- images/
  - churn_distribution.png
  - age_vs_churn.png
  - total_purchase_vs_churn.png
  - years_vs_churn.png
  - num_sites_vs_churn.png
  - account_manager_vs_churn.png
  - confusion_matrix.png
  - roc_curve.png
  - feature_importance.png

- models/
  - random_forest_churn_model.pkl

- notebooks/
  - churn_analysis.ipynb

- README.md
- requirements.txt
- .gitignore

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/TharushiKarunarathne/customer-churn-risk-analysis-and-prediction.git
```

### 2. Navigate to the project folder

```bash
cd customer-churn-risk-analysis-and-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/churn_analysis.ipynb
```

Run all cells from top to bottom.

---

## Requirements

The required Python libraries are listed in `requirements.txt`.

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

---

## Results

The project successfully built and compared machine learning models for customer churn prediction.

The Random Forest model was selected as the final model because it performed well and also provided feature importance values. This helped explain which customer features were useful for predicting churn.

The final trained model was saved and used to predict churn risk for new customers.

---

## Business Insights

Based on the analysis and model results:

- Customer churn can be predicted using basic customer information.
- Customers with a higher number of sites may show stronger churn risk.
- Customer tenure and purchase behavior are useful signals for churn prediction.
- Feature importance helps identify which customer characteristics are connected to churn.
- The model can help businesses take early action by identifying customers who may need additional support.

---

## Author

**Tharushi Karunarathne**

