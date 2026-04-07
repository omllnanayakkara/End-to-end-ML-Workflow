import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
from datetime import datetime

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_detection_experiment")

def load_and_preprocess_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Preprocess data
    encoder = LabelEncoder()
    train_df["merchant_category"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_category"] = encoder.transform(test_df["merchant_category"])

    print(f"Merchant category mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # Prepare features and labels
    X_train = train_df.drop("is_fraud", axis=1)
    y_train = train_df["is_fraud"]
    X_test = test_df.drop("is_fraud", axis=1)
    y_test = test_df["is_fraud"]

    return X_train, y_train, X_test, y_test, encoder

def train_model(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
):
    X_train, y_train, X_test, y_test, encoder = load_and_preprocess_data()

    with mlflow.start_run():
         run_name = f"rf_est{n_estimators}_depth{max_depth}_{datetime.now().strftime('%H%M%S')}"
         mlflow.set_tag("mlflow.runName", run_name)

         mlflow.log_param("n_estimators", n_estimators)
         mlflow.log_param("max_depth", max_depth)
         mlflow.log_param("min_samples_split", min_samples_split)
         mlflow.log_param("min_samples_leaf", min_samples_leaf)
         mlflow.log_param("model_type", "RandomForestClassifier")

         mlflow.log_param("training_samples", len(X_train))
         mlflow.log_param("test_samples", len(X_test))
         mlflow.log_param("fraud_ratio", y_train.mean())
         mlflow.log_param("n_features", X_train.shape[1])

         print(f"\nTraining model: n_estimators={n_estimators}, max_depth={max_depth}")
         model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
         )
         model.fit(X_train, y_train)

         for dataset_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, zero_division=0)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            roc_auc= roc_auc_score(y, y_prob)

            mlflow.log_metric(f"{dataset_name}_accuracy", acc)
            mlflow.log_metric(f"{dataset_name}_f1_score", f1)
            mlflow.log_metric(f"{dataset_name}_precision", precision)
            mlflow.log_metric(f"{dataset_name}_recall", recall)
            mlflow.log_metric(f"{dataset_name}_roc_auc", roc_auc)

            print(f"  {dataset_name.upper()} - Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

         for name, importance in sorted(
            zip(X_train.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
         ):
            mlflow.log_metric(f"feature_importance_{name}", importance)

         mlflow.sklearn.log_model(
             sk_model=model,
             artifact_path="model",
             registered_model_name="fraud-detection-model",
             input_example=X_train.iloc[:5]
         )

         with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
            mlflow.log_artifact("encoder.pkl")
        
         # Get the run ID for reference
         run_id = mlflow.active_run().info.run_id
         print(f"\nMLflow Run ID: {run_id}")
         print(f"View this run: http://localhost:5000/#/experiments/1/runs/{run_id}")

def run_experiment_sweep():
    """
    Run multiple experiments with different hyperparameters.
    
    This demonstrates how MLflow helps compare different configurations.
    """
    print("="*60)
    print("RUNNING HYPERPARAMETER EXPERIMENT SWEEP")
    print("="*60)
    
    # Define different configurations to try
    experiments = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
    ]
    
    for i, params in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_model(**params)
    
    print("\n" + "="*60)
    print("EXPERIMENT SWEEP COMPLETE!")
    print("="*60)
    print("\nView all experiments at: http://localhost:5000")
    print("Compare runs to find the best hyperparameters!")

if __name__ == "__main__":
    run_experiment_sweep()