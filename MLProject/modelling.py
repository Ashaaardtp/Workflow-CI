import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import sys

def main():
    try:
        # Set up MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("CI_Retraining")
        
        # Load data
        data_dir = "california_housing_preprocessed"
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' not found")
            sys.exit(1)
        
        print(f"Loading data from {data_dir}...")
        X_train = pd.read_csv(os.path.join(data_dir, "X_train_preprocessed.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test_preprocessed.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
        
        print(f"Data loaded successfully")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Start MLflow run
        with mlflow.start_run(run_name="CI_Run") as run:
            print(f"MLflow run ID: {run.info.run_id}")
            
            # Enable autolog
            mlflow.sklearn.autolog()
            
            # Train model
            print("Training RandomForestRegressor...")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
            
            # Log metrics manually
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Model logged successfully to MLflow")
            print(f"Run completed: {run.info.run_id}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
