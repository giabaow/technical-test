import os
import json
import logging
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix
import mlflow


test_path = os.getenv("TEST_PATH", "/data/processed/test.csv")
model_path = os.getenv("MODEL_PATH", "/data/model/model.joblib")
result_dir = os.getenv("RESULTS_DIR", "/data/results")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
experiment = os.getenv("MLFLOW_EXPERIMENT", "aumovio-homework")


def get_mlflow():
    if not mlflow_uri:
        return None
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        return mlflow
    except ImportError:
        return None
    

def evaluate():
    pipe = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X, y = df["Text"].astype(str), df["language"]

    print("Evaluating on %d samples", len(X))
    y_pred = pipe.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred, output_dict=True)

    print("Test accuracy : %.4f", acc)
    print("Weighted F1   : %.4f", f1)
    print("\n%s", classification_report(y, y_pred))

    # persist results
    os.makedirs(result_dir, exist_ok=True)
    results = {"accuracy": round(acc, 4),
               "weighted_f1": round(f1,  4),
               "classification_report": report,}
    output_path = os.path.join(result_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Result saved")

    # MLflow logging
    tracker = get_mlflow()
    if tracker:
        with tracker.start_run():
            tracker.log_metric("test_accuracy", acc)
            tracker.log_metric("test_weighted_f1", f1)
            tracker.log_artifact(output_path)

    print("Evaluation step complete")


if __name__ == "__main__":
    evaluate()