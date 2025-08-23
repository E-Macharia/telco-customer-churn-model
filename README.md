# Telecom Customer Churn Prediction

This project builds and evaluates machine learning models to predict customer churn for a telecommunications company. The goal is to identify customers likely to leave, enabling proactive retention strategies.

## Features

- **Exploratory Data Analysis (EDA):** Data cleaning, visualization, and feature engineering.
- **Model Building:** Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM.
- **Model Evaluation:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
- **Model Optimization:** Hyperparameter tuning using GridSearchCV.
- **Deployment:** Trained model saved for inference.

## Dataset

- [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/blastchar/telco-customer-churn)
- Features include customer demographics, account information, and service usage.

## Project Structure

```
Telecommunication_Churn_Model/
│
├── churn_model.ipynb         # Main Jupyter notebook with code and analysis
├── requirements.txt          # Python dependencies
├── logreg_model.joblib       # Saved trained model (after running notebook)
└── README.md                 # Project documentation
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/E-Macharia/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   Open `churn_model.ipynb` in Jupyter Notebook or VS Code and run all cells.

## Usage

- The notebook walks through EDA, feature engineering, model training, evaluation, and saving the best model.
- The final model is saved as `logreg_model.joblib` for deployment or inference.

## Requirements

- Python 3.7+
- See `requirements.txt` for package details.

## License

This project is for Practice purposes.

---
