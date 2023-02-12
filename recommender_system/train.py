import os
import datetime
import luigi
import json

import pandas as pd
import numpy as np

from surprise import SVD, NMF, SVDpp, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
from surprise import dump

import math
from sklearn.metrics import mean_squared_error, median_absolute_error

from .dataset import B2WPrepareDataset


DATA_FILES = {
    "without": {
        "output_path": os.path.join(os.getcwd(), "b2w/dataset"),
        "train_file": "train_rec.csv",
        "test_file": "test.csv",
    },
    "bert_timbau": {
        "output_path": os.path.join(
            os.getcwd(),
            "b2w/data_with_embeddings/bert-timbau-base-finetuning-sentiment-model-v0",
        ),
        "train_file": "train_rec.parquet",
        "test_file": "test.parquet",
    },
}


class TrainRS(luigi.Task):
    output_path: str = luigi.Parameter(default=os.path.join(os.getcwd(), "b2w/models"))
    sentiment_model: str = luigi.Parameter(default="without")
    recommender_model: str = luigi.Parameter(default="without")
    beta: float = luigi.FloatParameter(default=0.0)

    def run(self):
        output_path = os.path.join(
            self.output_path,
            str(
                self.recommender_model
                + "_"
                + self.sentiment_model
                + "_"
                + str(self.beta)
            ),
        )

        os.makedirs(output_path, exist_ok=True)

        print("---------- Generate Dataset")
        dataset = yield B2WPrepareDataset(
            DATA_FILES[self.sentiment_model]["output_path"],
            DATA_FILES[self.sentiment_model]["train_file"],
            DATA_FILES[self.sentiment_model]["test_file"],
        )

        train_df, test_df = self.load_data(
            dataset["train_df"].path, dataset["test_df"].path
        )

        print("---------- Trainig Model ", self.sentiment_model, self.recommender_model)
        algo, result = self.training(train_df, test_df)
        self.save_model(algo, output_path)

        with open(os.path.join(output_path, "results.json"), "w") as f:
            json.dump(result, f)

    def load_data(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if self.sentiment_model == "without":
            print("Without sentiment")
            train_df["rating"] = train_df["overall_rating"]
            test_df["rating"] = test_df["overall_rating"]
            test_df = test_df[["reviewer_id", "product_id", "rating"]]

        elif self.sentiment_model == "bert_timbau":
            print("With sentiment")

            train_df["rating"] = train_df["overall_rating"]
            test_df["rating"] = test_df["overall_rating"]
            test_df = test_df[["reviewer_id", "product_id", "rating"]]

            def set_range(x):
                if x >= 0 and x < 0.2:
                    return 1
                if x >= 0.2 and x < 0.4:
                    return 2
                if x >= 0.4 and x < 0.6:
                    return 3
                if x >= 0.6 and x < 0.8:
                    return 4
                if x >= 0.8 and x <= 1.0:
                    return 5

            mean_scores = (
                train_df.groupby(["product_id"]).mean()["scores"].reset_index()
            )
            mean_scores["sentiment_rating"] = mean_scores["scores"].apply(
                lambda x: set_range(x)
            )
            test_df = test_df.merge(mean_scores, on="product_id", how="left")
            print(test_df.sentiment_rating.isna().sum())
            test_df = test_df[
                ["reviewer_id", "product_id", "rating", "sentiment_rating"]
            ]

        reader = Reader(rating_scale=(1, 5))
        # The columns must correspond to user id, item id and ratings (in that order).
        train_data = Dataset.load_from_df(
            train_df[["reviewer_id", "product_id", "rating"]], reader
        )

        return (
            train_data,
            test_df,
        )

    def training(self, data, test_df):
        algo, params = self.get_model(data)

        kf = KFold(n_splits=5)
        for i, (trainset_cv, evalset_cv) in enumerate(kf.split(data)):
            print("fold number", i + 1)
            algo.fit(trainset_cv)

            print("Evaluation,", end="  ")
            predictions = algo.test(evalset_cv)
            accuracy.rmse(predictions, verbose=True)

            print("Train,", end=" ")
            predictions = algo.test(trainset_cv.build_testset())
            accuracy.rmse(predictions, verbose=True)

        predictions = algo.test(test_df[["reviewer_id", "product_id", "rating"]].values)
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(
            {"uid": "reviewer_id", "iid": "product_id"}, axis=1
        )

        if self.sentiment_model == "without":
            predictions = predictions.rename({"est": "final_rating"}, axis=1)

        if self.sentiment_model == "bert_timbau":
            predictions = predictions.merge(
                test_df[["reviewer_id", "product_id", "rating", "sentiment_rating"]],
                on=["product_id", "reviewer_id"],
                how="left",
            ).dropna(subset=["sentiment_rating"])

            predictions["final_rating"] = predictions[
                ["est", "sentiment_rating"]
            ].apply(
                lambda x: (self.beta * x["est"])
                + ((1 - self.beta) * x["sentiment_rating"]),
                axis=1,
            )

        rmse = math.sqrt(mean_squared_error(predictions.r_ui, predictions.final_rating))
        mse = mean_squared_error(predictions.r_ui, predictions.final_rating)
        mae = median_absolute_error(predictions.r_ui, predictions.final_rating)

        print("RMSE ", rmse)
        print("MSE ", mse)
        print("MAE ", mae)

        result = {
            "recommender_model": self.recommender_model,
            "sentiment_model": self.sentiment_model,
            "params": params,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
        }

        return algo, result

    def get_model(self, data):
        if self.recommender_model == "svd":
            return self.get_svd(data)
        elif self.recommender_model == "svdpp":
            return self.get_svdpp(data)
        elif self.recommender_model == "nmf":
            return self.get_nmf(data)
        else:
            raise "Invalid recommender model"

    def get_svd(self, data):
        # param_grid = {
        #     "n_factors": [10, 100, 500],
        #     "n_epochs": [20, 50, 100],
        #     "lr_all": [0.001, 0.005, 0.02, 0.1],
        #     "reg_all": [0.005, 0.02, 0.1],
        # }
        # gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
        # gs.fit(data)
        # params = gs.best_params["rmse"]
        svd = SVD()

        return svd, {}

    def get_svdpp(self, data):
        # param_grid = {
        #     "n_factors": [10, 100, 500],
        #     "n_epochs": [20, 50, 100],
        #     "lr_all": [0.001, 0.005, 0.02, 0.1],
        #     "reg_all": [0.005, 0.02, 0.1],
        # }
        # gs = GridSearchCV(SVDpp, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
        # gs.fit(data)
        # params = gs.best_params["rmse"]
        svdpp = SVDpp()

        return svdpp, {}

    def get_nmf(self, data):
        # param_grid = {
        #     "n_factors": [10, 100, 500],
        #     "n_epochs": [20, 50, 100],
        #     "lr_bu": [0.001, 0.005, 0.02, 0.1],
        #     "lr_bi": [0.001, 0.005, 0.02, 0.1],
        #     "reg_pu": [0.005, 0.02, 0.1],
        #     "reg_qi": [0.005, 0.02, 0.1],
        # }
        # gs = GridSearchCV(NMF, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
        # gs.fit(data)
        # params = gs.best_params["rmse"]
        nmf = NMF()

        return nmf, {}

    def save_model(self, algo, path):
        save_path = os.path.join(path, str(self.recommender_model + ".pkl"))
        dump.dump(save_path, algo=algo)
