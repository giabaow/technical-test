import pandas as pd
import joblib
import json
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import mlflow


train_path  = os.getenv("TRAIN_PATH","/data/processed/train.csv")
model_dir   = os.getenv("MODEL_DIR", "/data/model")
mlflow_uri  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
experiment  = os.getenv("MLFLOW_EXPERIMENT", "aumovio-homework")
random_seed = int(os.getenv("SEED","42"))


# TF-IDF
char_max_features = int(os.getenv("TFIDF_CHAR_MAX_FEATURES", "150000"))
word_max_features = int(os.getenv("TFIDF_MAX_FEATURES", "50000"))
ngram_char_max = int(os.getenv("TFIDF_NGRAM_CHAR_MAX","4"))
ngram_word_max = int(os.getenv("TFIDF_NGRAM_MAX", "2"))

# Classifiers
C = float(os.getenv("LR_C", "5.0"))
svc_C = float(os.getenv("SVC_C","1.0"))
max_iter = int(os.getenv("LR_MAX_ITER","1000"))

CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))


# MLflow 
def get_mlflow():
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        return mlflow
    except Exception:
        return None


# Feature extraction 
def build_features():
    char_tfidf = TfidfVectorizer(analyzer="char_wb",
                                 ngram_range=(1, ngram_char_max),
                                 max_features=char_max_features,
                                 sublinear_tf=True,
                                 strip_accents=None,
                                 min_df=2,)
    
    word_tfidf = TfidfVectorizer(analyzer="word",
                                 ngram_range=(1, ngram_word_max),
                                 max_features=word_max_features,
                                 sublinear_tf=True,
                                 min_df=2,)
    
    return FeatureUnion([("char", char_tfidf), ("word", word_tfidf)])


# Models 
def get_candidates():
    features = build_features()

    lr = LogisticRegression(C=C, max_iter=max_iter, solver="saga", random_state=random_seed,)

    svc = CalibratedClassifierCV(LinearSVC(C=svc_C, max_iter=max_iter, random_state=random_seed), cv=3,)

    sgd = SGDClassifier(loss="modified_huber", alpha=1e-4,max_iter=200, random_state=random_seed,)

    ensemble = VotingClassifier(estimators=[("lr", lr), ("svc", svc), ("sgd", sgd)],voting="soft",)

    return {"logistic_regression": Pipeline([("features", build_features()), ("clf", lr)]),
            "linear_svc": Pipeline([("features", build_features()), ("clf", svc)]),
            "sgd": Pipeline([("features", build_features()), ("clf", sgd)]),
            "ensemble": Pipeline([("features", build_features()), ("clf", ensemble)]),}


# Train 
def train():
    df = pd.read_csv(train_path)
    X, y = df["Text"].astype(str), df["language"]

    tracker = get_mlflow()
    candidates = get_candidates()

    # cross-validation
    print(f"Evaluating {len(candidates)} models with {CV_FOLDS}-fold CV...\n")
    cv_results = {}

    for name, pipe in candidates.items():
        print(f"  Evaluating: {name} ...")
        scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = scores.mean(), scores.std()
        cv_results[name] = {"mean": mean_acc, "std": std_acc}
        print(f"CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        if tracker:
            with tracker.start_run(run_name=name):
                tracker.log_param("model", name)
                tracker.log_metric("cv_mean_accuracy", mean_acc)
                tracker.log_metric("cv_std_accuracy", std_acc)

    # the best model 
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    best_score = cv_results[best_name]["mean"]

    # Refit best model
    best_pipe = candidates[best_name]
    best_pipe.fit(X, y)

    train_acc = accuracy_score(y, best_pipe.predict(X))
    report = classification_report(y, best_pipe.predict(X))
    print(f"Train accuracy (full data): {train_acc:.4f}")
    print(report)

    # Save the best model 
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"Model saved: {model_path}")

    labels = sorted(best_pipe.classes_.tolist())
    meta   = {"best_model":   best_name,
              "cv_accuracy":  best_score,
              "train_accuracy": train_acc,
              "labels":       labels,
              "num_classes":  len(labels),
              "cv_results":   cv_results,}
    
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    if tracker:
        with tracker.start_run(run_name=f"BEST_{best_name}"):
            tracker.log_param("best_model", best_name)
            tracker.log_metric("cv_accuracy", best_score)
            tracker.log_metric("train_accuracy", train_acc)
            tracker.log_artifact(model_path)
            tracker.log_artifact(meta_path)
    
    print("Complete")


if __name__ == "__main__":
    train()
