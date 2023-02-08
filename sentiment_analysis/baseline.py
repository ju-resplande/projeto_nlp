from typing import List
import pickle
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd

from utils import preprocess_sentence
from model import AbsModel

DATA_PATH = "algum lugar"

# TODO: Colocar paralelismo
# TODO: Testar


class Baseline(AbsModel):
    def __init__(self) -> None:
        self.embedder = TfidfVectorizer(preprocessor=preprocess_sentence)
        self.model = LogisticRegressionCV(cv=5, random_state=0)

    def encode(self, inputs: List[str]) -> List[float]:
        encoded = self.embedder.transform(inputs)

        return encoded

    def get_logits(self, inputs: List[str]) -> List[float]:
        X = self.encode(inputs)
        logits = self.model.predict_log_proba(X)

        return logits


def main():
    data = dict()
    for split in ["train", "test"]:
        data[split] = pd.read_csv(os.path.append(DATA_PATH, f"{split}.csv"))

    baseline = Baseline()
    baseline.embedder.fit_transform(data["train"])
    baseline.model.fit(baseline.encode(data["train"]), data["train"]["label"])

    predictions = baseline.model.predict(baseline.encode(data["test"]["text"]))
    with open(os.path.append("predictions", "tfidf.json"), "w"):
        json.dump(predictions)

    with open(os.path.append("output", "baseline.pkl"), "wb") as f:
        pickle.dump(baseline, f)


if __name__ == "__main__":
    main()
