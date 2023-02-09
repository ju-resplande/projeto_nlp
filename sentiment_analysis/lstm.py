# based on https://github.com/lukysummer/Movie-Review-Sentiment-Analysis-LSTM-Pytorch/blob/master/sentiment_analysis_LSTM.py
from typing import List, Final, Tuple
import pickle
import json
import csv
import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from utils import preprocess_sentence
from model import AbsModel

DATA_PATH = "algum lugar"

# TODO: ver pad_idx
# TODO: colocar loguru
# TODO: colocar tensorboard
# TODO: Colocar paralelismo
# TODO: qual embedding pegar?
# TODO: testar


class Word2Vec:
    pad_idx: Final = 0

    def __init__(self, vector_file: str) -> None:
        self.vocab, self.embedding_matrix = self.read_embeddings(vector_file)

        self.word2idx: dict = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word: dict = self.vocab

        self.embedding_matrix: np.ndarray = np.array(self.embedding_matrix)

    @staticmethod
    def read_embeddings(vector_file: str) -> None:
        vocab, embedding_matrix = list(), list()
        with open(vector_file) as f:
            csv_reader = csv.reader(f, delimiter="\t")

        for row in tqdm(csv_reader):
            vocab.append(row[0])
            embedding_matrix.append(row[1:])

        return vocab, embedding_matrix

    def get_idx(self, inputs: List[str], max_seq_len: int) -> List[float]:
        inputs_idx = list()
        for string in tqdm(inputs):
            string = preprocess_sentence(string)
            words = " ".split(string)

            input_idx = [self.word2idx.get(w, Word2Vec.pad_idx) for w in words]

            if len(input_idx) < max_seq_len:
                input_idx += (max_seq_len - len(input_idx)) * [Word2Vec.pad_idx]

            inputs_idx.append(input_idx)

        inputs_idx = np.array(inputs_idx)
        return inputs_idx


class LSTM(nn.Module, AbsModel):
    config = {
        "n_hidden": 512,
        "n_layers": 2,
        "dropout_p": 0.5,
        "lr": 0.001,
        "clip": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(
        self, embedding_matrix: np.ndarray, n_class: int, max_seq_len: int, device: str
    ) -> None:
        self.config = LSTM.config.copy()
        self.config["max_seq_len"] = max_seq_len
        self.config["n_class"] = n_class
        self.config["device"] = device

        self.config["n_vocab"] = embedding_matrix.shape[0]
        self.config["n_embed"] = embedding_matrix.shape[1]

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])

        self.embed = nn.Embedding(
            self.config["n_vocab"],
            self.config["n_embed"],
            input_length=self.config["max_seq_len"],
            weights=[embedding_matrix],
        )
        self.lstm = nn.LSTM(
            self.config["n_embed"],
            self.config["n_hidden"],
            self.config["n_layers"],
            dropout=self.config["dropout_p"],
            batch_first=True,
        )
        self.dropout = (nn.Dropout(self.config["dropout_p"]),)
        self.fc = nn.Linear(self.config["n_hidden"], self.config["n_class"])
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, batch_size) -> Tuple[torch.Tensor]:
        embedded = self.embed(inputs)

        output, hidden = self.lstm(embedded)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.n_hidden)

        output = self.fc(output)

        output = self.sigmoid(output)
        output = output.view(batch_size, -1)
        output = output[:, -1]

        return output, hidden

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        h = (
            weights.new(self.n_layers, batch_size, self.n_hidden)
            .zero_()
            .to(self.config["device"]),
            weights.new(self.n_layers, batch_size, self.n_hidden)
            .zero_()
            .to(self.config["device"]),
        )

        return h

    def do_training(
        self,
        n_epochs: int,
        batch_size: int,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        print_every: int,
    ):
        step = 0
        for epoch in range(n_epochs):
            hidden = self.init_hidden(batch_size)

            for inputs, labels in train_data_loader:
                step += 1
                inputs, labels = inputs.to(self.config["device"]), labels.to(
                    self.config["device"]
                )

                hidden = (h.data for h in hidden)
                self.zero_grad()

                output, hidden = self(inputs)
                loss = self.criterion(output.squeeze(), labels.float())

                loss.backward()
                nn.utils.clip_grad_norm(self.parameters(), self.config["clip"])
                self.optimizer.step()

                if (step % print_every) == 0:
                    step += 1
                    self.do_validation(batch_size, val_data_loader)
                    self.train()

                print(
                    "Epoch: {}/{}".format((epoch + 1), n_epochs),
                    "Step: {}".format(step),
                    "Training Loss: {:.4f}".format(loss.item()),
                    "Validation Loss: {:.4f}".format(),
                )

    def do_validation(self, batch_size: int, data_loader: DataLoader):
        self.eval()

        hidden = self.init_hidden(batch_size)
        losses = list()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.config["device"]), labels.to(
                self.config["device"]
            )

            hidden = tuple([h.data for h in hidden])

            output, _ = self(inputs)
            loss = self.criterion(output.squeeze(), labels.float())

            losses.append(loss.item())

        loss = np.mean(losses)

        return loss

    def do_test(self, batch_size: int, data_loader: DataLoader):
        total_preds = []
        total_labels = []

        self.eval()
        hidden = self.init_hidden(batch_size)
        losses = ()
        for inputs, labels in data_loader:
            hidden = tuple([each.data for each in hidden])
            output, hidden = self(inputs, hidden)
            loss = self.criterion(output, labels)
            losses.append(loss.item())

            preds = torch.round(output.squeeze())
            total_preds.extend(preds)
            total_labels.extend(labels)

        return total_labels, total_preds

    def do_predict(
        self,
        inputs: List[List[float]],
    ):
        self.eval()

        hidden = self.init_hidden(1)
        output, hidden = self(inputs, hidden)

        return output, hidden


def main():
    max_seq_len = 80
    print_every = 3
    batch_size = 8
    n_epochs = 4

    ## Load, preprocess and embed data
    embedder = Word2Vec("word2vec.kv")

    data_loader = dict()
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.append(DATA_PATH, f"{split}.csv")).to_numpy()
        X = df[:-1]
        X = embedder.get_idx(X, max_seq_len)

        Y = df[0]
        n_class = Y.unique().shape[0]

        data_loader[split] = DataLoader(TensorDataset(X, Y), batch_size=batch_size)

    ## Train model
    embedding_matrix = embedder.embedding_matrix
    model = LSTM(embedding_matrix, n_class, max_seq_len)
    model.do_training(
        n_epochs, batch_size, data_loader["train"], data_loader["dev"], print_every
    )

    ## Test model
    Y = embedder.get_idx()
    labels, preds = model.do_test(batch_size, data_loader["test"])
    print(classification_report(labels, preds))


if __name__ == "__main__":
    main()
