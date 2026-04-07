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
    classification_report
)

from matplotlib import pyplot as plt

def main(): 
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

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]:,} (correctly identified legitimate)")
    print(f"  False Positives: {cm[0][1]:,} (legitimate flagged as fraud)")
    print(f"  False Negatives: {cm[1][0]:,} (fraud missed - DANGEROUS!)")
    print(f"  True Positives:  {cm[1][1]:,} (correctly caught fraud)")

    # Feature importance
    print("\nFeature Importance:")
    for name, importance in sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {name}: {importance:.4f}")
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

    # Save the model and encoder together
    print("\nSaving model to models/model.pkl...")
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, encoder), f)
    
    print("\nModel trained and saved successfully!")


if __name__ == "__main__":    main()