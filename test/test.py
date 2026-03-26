import json
import os
import sys
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_preparation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))

from preparation import clean_text
from train import get_candidates, build_features


sample_texts = [
    ("Hello how are you","english"),
    ("Good morning have a nice day","english"),
    ("The weather is lovely today","english"),
    ("Bonjour comment va","french"),
    ("Je suis content","french"),
    ("Bonne journee tous","french"),
    ("Hola como estas","spanish"),
    ("Buenos dias todos","spanish"),
    ("Me alegra verte nuevo","spanish"),
]


# data preperation tests
class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_removes_urls(self):
        assert "http" not in clean_text("visit https://example.com today")

    def test_removes_html_tags(self):
        assert "<b>" not in clean_text("<b>hello</b>")

    def test_removes_punctuation(self):
        result = clean_text("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_returns_string(self):
        assert isinstance(clean_text("hello"), str)

    def test_handles_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)


# train tests
class TestPipeline:
    @pytest.fixture
    def trained_pipe(self) -> Pipeline:
        df   = pd.DataFrame(sample_texts, columns=["Text", "language"])
        # use the best candidate for testing
        candidates = get_candidates()
        pipe = candidates["logistic_regression"]   # fastest
        pipe.fit(df["Text"], df["language"])
        return pipe

    def test_pipeline_is_sklearn(self, trained_pipe):
        assert isinstance(trained_pipe, Pipeline)

    def test_predicts_known_languages(self, trained_pipe):
        pred = trained_pipe.predict(["Hello world"])
        assert pred[0] == "english"

    def test_predict_proba_sums_to_one(self, trained_pipe):
        proba = trained_pipe.predict_proba(["Bonjour"])[0]
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_top_class_matches_predict(self, trained_pipe):
        text  = ["Hola amigo"]
        pred  = trained_pipe.predict(text)[0]
        proba = trained_pipe.predict_proba(text)[0]
        top   = trained_pipe.classes_[proba.argmax()]
        assert pred == top

    def test_model_serialisation(self, trained_pipe, tmp_path):
        path = tmp_path / "model.joblib"
        joblib.dump(trained_pipe, path)
        loaded = joblib.load(path)
        assert loaded.predict(["Hello"])[0] == trained_pipe.predict(["Hello"])[0]

    def test_classes_count(self, trained_pipe):
        assert len(trained_pipe.classes_) == 3

    def test_meta_json_structure(self, trained_pipe, tmp_path):
        labels = sorted(trained_pipe.classes_.tolist())
        meta   = {
            "best_model":     "logistic_regression",
            "cv_accuracy":    0.95,
            "train_accuracy": 0.99,
            "labels":         labels,
            "num_classes":    len(labels),
        }
        meta_path = tmp_path / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        with open(meta_path) as f:
            loaded = json.load(f)

        assert "labels" in loaded
        assert "best_model" in loaded
        assert "num_classes" in loaded
        assert loaded["num_classes"] == 3


# API tests 
class TestAPI:
    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        df   = pd.DataFrame(sample_texts, columns=["Text", "language"])
        candidates = get_candidates()
        pipe = candidates["logistic_regression"]
        pipe.fit(df["Text"], df["language"])

        model_path = str(tmp_path / "model.joblib")
        meta_path  = str(tmp_path / "meta.json")

        joblib.dump(pipe, model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "labels":      pipe.classes_.tolist(),
                "num_classes": 3,
                "best_model":  "logistic_regression",
                "params":      {}
            }, f)

        monkeypatch.setenv("MODEL_PATH", model_path)
        monkeypatch.setenv("META_PATH",  meta_path)

        import importlib
        import deploy.app as app_module
        importlib.reload(app_module)

        from fastapi.testclient import TestClient
        self.client = TestClient(app_module.app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_languages_endpoint(self):
        r = self.client.get("/languages")
        assert r.status_code == 200
        assert "languages" in r.json()

    def test_predict_english(self):
        r = self.client.post("/predict", json={"text": "Hello how are you today"})
        assert r.status_code == 200
        body = r.json()
        assert body["language"] == "english"
        assert 0 <= body["confidence"] <= 1
        assert len(body["top_3"]) == 3

    def test_predict_empty_text(self):
        r = self.client.post("/predict", json={"text": "   "})
        assert r.status_code == 422

    def test_predict_returns_top3_sorted(self):
        r = self.client.post("/predict", json={"text": "Bonjour tout le monde"})
        assert r.status_code == 200
        top3        = r.json()["top_3"]
        confidences = [item["confidence"] for item in top3]
        assert confidences == sorted(confidences, reverse=True)