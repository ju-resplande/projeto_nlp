import os
import pandas as pd
import numpy as np
import luigi

from sklearn import preprocessing


class B2WPrepareDataset(luigi.Task):
    output_path: str = luigi.Parameter(default=os.path.join(os.getcwd(), "b2w/dataset"))
    train_file: str = luigi.Parameter(default="train_rec.csv")
    test_file: str = luigi.Parameter(default="test_rec.csv")

    def output(self):
        return {
            "train_df": luigi.LocalTarget(
                os.path.join(self.output_path, "train_rec_processed.csv")
            ),
            "test_df": luigi.LocalTarget(
                os.path.join(self.output_path, "test_rec.csv")
            ),
        }

    def run(self):
        print("---------- Prepare Dataset")

        if self.train_file.split(".")[-1] == "csv":
            train_df = pd.read_csv(os.path.join(self.output_path, self.train_file))
        elif self.train_file.split(".")[-1] == "parquet":
            train_df = pd.read_parquet(
                os.path.join(self.output_path, self.train_file), engine="pyarrow"
            )
        else:
            raise "Unsupported file extension"

        if self.test_file.split(".")[-1] == "csv":
            test_df = pd.read_csv(os.path.join(self.output_path, self.test_file))
        elif self.test_file.split(".")[-1] == "parquet":
            test_df = pd.read_parquet(
                os.path.join(self.output_path, self.test_file), engine="pyarrow"
            )
        else:
            raise "Unsupported file extension"

        train_df = train_df.dropna(subset=["reviewer_id", "product_id"])
        print("Train data with shape: ", train_df.shape)
        print("Test data with shape: ", test_df.shape)

        all_products = np.concatenate(
            (train_df["product_id"].unique(), test_df["product_id"].unique())
        )
        product_encoder = preprocessing.LabelEncoder().fit(all_products)

        train_df["product_id"] = product_encoder.transform(
            train_df["product_id"].values
        )
        test_df["product_id"] = product_encoder.transform(test_df["product_id"].values)

        all_reviewers = np.concatenate(
            (train_df["reviewer_id"].unique(), test_df["reviewer_id"].unique())
        )
        reviewer_encoder = preprocessing.LabelEncoder().fit(all_reviewers)

        train_df["reviewer_id"] = reviewer_encoder.transform(
            train_df["reviewer_id"].values
        )
        test_df["reviewer_id"] = reviewer_encoder.transform(
            test_df["reviewer_id"].values
        )

        train_df.to_csv(self.output()["train_df"].path, index=False)
        test_df.to_csv(self.output()["test_df"].path, index=False)
