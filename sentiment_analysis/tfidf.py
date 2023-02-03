import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import pandas as pd

from utils import preprocess_sentence

DATA_PATH = "algum lugar"
OUT_PATH = "output"

def main():
    train = pd.read_csv(os.path.append(DATA_PATH, "train.csv"))
    test = pd.read_csv(os.path.append(DATA_PATH, "test.csv"))

    train["label"] = train["label"].map({"positive": 1, "negative": 0}) 
    test["label"] = test["label"].map({"positive": 1, "negative": 0}) 

    train_corpus = ""
    for example in tqdm(train["text"]):
        train_corpus += "\n" + example

    prepro_with_stem = lambda text: preprocess_sentence(text, do_stemming=True)
    vectorizer = TfidfVectorizer(preprocessor=prepro_with_stem)
    vectorizer.fit_transform(train_corpus)

    classifier = LogisticRegressionCV(cv=5, random_state=0)
    classifier.fit(train["text"], train["label"])
    predictions = classifier.predict(test["text"])

    with open(os.path.append(OUT_PATH, "tfidf.json"), "w"):
        json.dump(predictions)


if __name__ == "__main__":
    main()
