import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'D:\Loan Predictor\data\loan_data.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Ensure column names are trimmed of whitespace
data.columns = data.columns.str.strip()

# Print column names for verification
print("Column Names After Preprocessing:")
print(data.columns)

# Verify exact column name and case sensitivity
target_column = 'loan_Status_ Rejected'  # Adjust according to your actual target column name
if target_column in data.columns:
    # Split the data
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)

    # Save the model and scaler
    joblib.dump(rf, 'loan_approval_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate models
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test_scaled)

    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'ROC AUC Score: {roc_auc}')

    print("Logistic Regression Performance:")
    evaluate_model(y_test, y_pred_log_reg)

    print("\nRandom Forest Performance:")
    evaluate_model(y_test, y_pred_rf)

    # Model interpretation using SHAP
# Assuming X_train_scaled, X_test_scaled are correctly defined and used for training and testing
# Train RandomForestClassifier (rf) and prepare for SHAP explanation

# Model interpretation using SHAP
    explainer = shap.Explainer(rf, X_train_scaled)
    shap_values = explainer(X_test_scaled, check_additivity=False)  # Disable additivity check

# Plot SHAP summary plot or continue with further analysis
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)


else:
    print(f"'{target_column}' column not found in dataset. Check column names and ensure it exists.")
