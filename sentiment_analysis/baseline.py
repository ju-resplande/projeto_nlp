import autosklearn.classification
from sklearn.metrics import Clas
import pandas as pd
import sklearn


export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

DATA_PATH = "b2w/dataset"

def main():
    data = dict()
    for split in ["train", "test"]:
        data[split] = pd.read_csv(os.path.append(DATA_PATH, f"{split}_sa.csv"))

    embedder = TfidfVectorizer(preprocessor=preprocess_sentence)
    embedder.fit_transform(data["train"]["review"])
    
    X, Y = dict(), dict()
    for split in ["train", "test"]: 
        X[split] = embedder.encode(data[split]["review"])
        Y[split] = data["rating"]

    automl = autosklearn.classification.AutoSklearnClassifier(n_jobs=12)
    automl.fit(X["train"], Y["train"])
    y_hat = automl.predict(X["test"])
    
    print(classification_report(Y["test"], y_hat))
    
    poT = cls.performance_over_time_
    poT.plot(
        x="Timestamp",
        kind="line",
        legend=True,
        title="Auto-sklearn accuracy over time",
        grid=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
