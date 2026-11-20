# Customer Churn Prediction

Predicts which customers are likely to leave (churn) using three machine learning models: **Logistic Regression, Random Forest and XGBoost**. The notebook walks through the full process step by step: combining the data, exploring it, preparing features, training the models and comparing their results.

## Dataset
[Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) on Kaggle (by muhammadshahidazeem). It comes as a training file and a testing file, which the notebook combines into one dataset of **505,206 customers and 12 columns**.

| Type | Columns |
|---|---|
| Numeric | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction |
| Categorical | Gender, Subscription Type, Contract Length |
| Target | Churn (1 = left, 0 = stayed) |

The CSV files are not included in this repository because of their size. Download them from the Kaggle link above.

## Method
1. Combine the training and testing files and remove incomplete rows
2. Explore the data (summary statistics, distributions, category counts)
3. One-hot encode the categorical columns and drop `CustomerID`
4. Split the data 70% train / 15% test / 15% validation (`random_state=42`)
5. Train the three models with default settings
6. Evaluate with accuracy, precision, recall and F1-score

## Results (validation set)
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Random Forest | 0.9347 | 0.8966 | 0.9977 | 0.9445 |
| XGBoost | 0.9332 | 0.8968 | 0.9944 | 0.9431 |
| Logistic Regression | 0.8463 | 0.8741 | 0.8457 | 0.8597 |

Random Forest and XGBoost perform almost the same and clearly beat Logistic Regression. Test-set scores are very close to the validation scores. Logistic Regression gives a `ConvergenceWarning` with default settings. Raising `max_iter` or scaling the features is a possible next step.

## How to run
1. Clone this repository: `git clone <(https://github.com/kriti191220/CustomerChurnPrediction)l>`
2. Install the requirements: `pip install -r requirements.txt`
3. Download the two CSV files from Kaggle and place them in the project folder
4. Open `Churn_Prediction_Step_by_Step.ipynb` and run all cells (update the file paths in Step 2 if your file names differ)

## Repository contents
- `Churn_Prediction_Step_by_Step.ipynb`: the full analysis and models
- `requirements.txt`: Python libraries needed

## Tools
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Author
Kriti Tandon · <(https://www.linkedin.com/in/kriti-tandon-b02381229/)>
