import mlflow
import mlflow.sklearn
import joblib
import datetime
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models import infer_signature

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the latest model from MLflow
MODEL_URI = "models:/linear_regression_model/latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Load original training dataset
df_original = pd.read_csv("../data/processed/2024/train.csv", parse_dates=['Trip_Start', 'Trip_End'])

# Load stored new data from API inputs
new_data_file = "../api/saved_inputs/prediction_inputs.csv"

if os.path.exists(new_data_file):
    df_new = pd.read_csv(new_data_file)
    print(f"Loaded {len(df_new)} new data points for retraining.")
else:
    df_new = None
    print("No new data found. Skipping retraining.")

# Sample 10% of original data for fairness
df_original = df_original.sample(frac=0.1, random_state=42)

# Feature Engineering
df_original['Trip_Duration'] = (df_original['Trip_End'] - df_original['Trip_Start']).dt.total_seconds()
df_original['Start_Hour'] = df_original['Trip_Start'].dt.hour
df_original['Start_DayOfWeek'] = df_original['Trip_Start'].dt.dayofweek

df_original = df_original.dropna(subset=['Trip_Duration', 'Year_of_Birth', 'Gender', 'Origin_Id', 'Destination_Id'])

# Merge new and original data if available
if df_new is not None:
    df_combined = pd.concat([df_original, df_new], ignore_index=True)
else:
    df_combined = df_original

# Define Features and Target
X = df_combined[['Year_of_Birth', 'Gender', 'Origin_Id', 'Destination_Id', 'Start_Hour', 'Start_DayOfWeek']]
y = df_combined['Trip_Duration']

# Ensure correct data types
X = X.astype({
    "Year_of_Birth": float,
    "Gender": int,
    "Origin_Id": int,
    "Destination_Id": int,
    "Start_Hour": int,
    "Start_DayOfWeek": int
})

# Preprocessing pipeline
categorical_features = ['Gender']
numerical_features = ['Year_of_Birth', 'Origin_Id', 'Destination_Id', 'Start_Hour', 'Start_DayOfWeek']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Evaluate current model performance
y_pred_existing = model.predict(X_test)
mae_existing = mean_absolute_error(y_test, y_pred_existing)

# Define degradation threshold
THRESHOLD = 5.0  # Adjust based on acceptable performance drop

# If performance degrades, retrain
if mae_existing > THRESHOLD:
    print(f"Performance degraded (MAE: {mae_existing:.4f}). Retraining model...")

    mlflow.set_experiment("trip_duration_prediction")

    with mlflow.start_run():
        # Train new model
        lr_pipeline.fit(X_train, y_train)
        y_pred_new = lr_pipeline.predict(X_test)

        # Compute new metrics
        mae_new = mean_absolute_error(y_test, y_pred_new)
        rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))

        # Infer schema
        signature = infer_signature(X_train, y_pred_new)

        # Log new metrics
        mlflow.log_metric("mae", mae_new)
        mlflow.log_metric("rmse", rmse_new)

        # Save new model
        mlflow.sklearn.log_model(lr_pipeline, "linear_regression_model", signature=signature)

        print(f"New model retrained and logged (MAE: {mae_new:.4f}).")

    # Delete old input data to avoid duplicate training
    os.remove(new_data_file)
    print("Old input data deleted.")
else:
    print(f"No retraining needed. Current MAE: {mae_existing:.4f} is within acceptable range.")
