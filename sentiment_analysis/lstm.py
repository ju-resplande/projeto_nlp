# based on https://github.com/lukysummer/Movie-Review-Sentiment-Analysis-LSTM-Pytorch/blob/master/sentiment_analysis_LSTM.py
from typing import List, Final, Tuple
import pickle
import json
import csv
import os

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from utils import preprocess_sentence
from model import AbsModel

DATA_PATH = "../data"

# TODO: ver pad_idx
# TODO: colocar loguru
# TODO: Colocar paralelismo
# TODO: qual embedding pegar?
# TODO: testar embedding

# TODO: testar modelo


class Word2Vec:
    pad_idx: Final = 0

    def __init__(self, vector_file: str) -> None:
        self.vocab, self.embedding_matrix = self.read_embeddings(vector_file)

        self.word2idx: dict = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word: dict = self.vocab

        #self.embedding_matrix: np.ndarray = self.embedding_matrix

    def read_embeddings(self, vector_file: str) -> None:
        vocab, embedding_matrix = list(), list()
        with open(vector_file) as f:
            vocab_size, embedding_size = f.readline().strip().split()
            self.embedding_matrix = np.empty((int(vocab_size), int(embedding_size)), dtype=np.float32)
            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in tqdm(enumerate(csv_reader)):
                vocab.append(row[0])
                self.embedding_matrix[i] = np.asarray(row[1:])
                #embedding_matrix.append(np.asarray(row[1:]))

        return vocab, self.embedding_matrix

    def get_idx(self, inputs: List[str], max_seq_len: int) -> List[float]:
        inputs_idx = list()
        for string in tqdm(inputs):
            string = preprocess_sentence(string)
            words = string.split(" ")

            input_idx = [self.word2idx.get(w, Word2Vec.pad_idx) for w in words]

            if len(input_idx) < max_seq_len:
                input_idx += (max_seq_len - len(input_idx)) * [Word2Vec.pad_idx]
            
            inputs_idx.append(np.asarray(input_idx[:max_seq_len]))

        inputs_idx = np.array(inputs_idx)
        return inputs_idx


class LSTM(nn.Module):
    config = {
        "n_hidden": 64,
        "n_layers": 2,
        "dropout_p": 0.5,
        "lr": 0.0001,
        "clip": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(
        self, embedding_matrix: np.ndarray, n_class: int, max_seq_len: int, device: str
    ) -> None:
        super(LSTM, self).__init__()
        self.config = LSTM.config.copy()
        self.config["max_seq_len"] = max_seq_len
        self.config["n_class"] = n_class
        self.config["device"] = device

        self.config["n_vocab"] = embedding_matrix.shape[0]
        self.config["n_embed"] = embedding_matrix.shape[1]

        self.embed = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_matrix),
            freeze=True
        )
        self.lstm = nn.LSTM(
            self.config["n_embed"],
            self.config["n_hidden"],
            self.config["n_layers"],
            dropout=self.config["dropout_p"],
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config["dropout_p"])
        self.fc = nn.Linear(self.config["n_hidden"], self.config["n_class"])
        self.sigmoid = nn.Sigmoid()
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        self.to(self.config["device"])

    def forward(self, inputs) -> Tuple[torch.Tensor]:
        batch_size = inputs.shape[0]
        embedded = self.embed(inputs)

        output, hidden = self.lstm(embedded)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.config["n_hidden"])

        output = self.fc(output)

        output = self.sigmoid(output)
        output = output.view(batch_size, -1)
        output = output[:, -1]

        return output, hidden

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        h = (
            weights.new(self.config["n_layers"], batch_size, self.config["n_hidden"])
            .zero_()
            .to(self.config["device"]),
            weights.new(self.config["n_layers"], batch_size, self.config["n_hidden"])
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
        writer: SummaryWriter,
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

                writer.add_scalar("Loss/train", loss.item(), step)

                if (step % print_every) == 0:
                    step += 1
                    val_loss, cls_report = self.do_validation(batch_size, val_data_loader)
                    writer.add_scalar("Loss/validation", val_loss.item(), step)
                    self.train()

                    print(
                        "Epoch: {}/{}".format((epoch + 1), n_epochs),
                        "Step: {}".format(step),
                        "Training Loss: {:.4f}".format(loss.item()),
                        "Validation Loss: {:.4f}".format(val_loss.item())
                    )
                    print(cls_report)

    def do_validation(self, batch_size: int, data_loader: DataLoader):
        self.eval()

        hidden = self.init_hidden(batch_size)
        losses = list()
        y_true = []
        y_pred = []
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.config["device"]), labels.to(
                self.config["device"]
            )

            hidden = tuple([h.data for h in hidden])

            output, _ = self(inputs)
            loss = self.criterion(output.squeeze(), labels.float())

            losses.append(loss.item())
            
            y_true.extend(labels.cpu().detach().numpy())
            y_pred.extend((output.squeeze().cpu().detach().numpy() > 0.5).astype(np.intc))
        
        cls_report = classification_report(y_true, y_pred)
            
        loss = np.mean(losses)

        return loss, cls_report

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
    
        loss = np.mean(losses)
        cls_report = classification_report(total_labels, total_preds)
        print(cls_report)

        return total_labels, total_preds, loss

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
    print_every = 100
    batch_size = 8
    n_epochs = 4

    writer = SummaryWriter(comment="LSTM_training")

    ## Load, preprocess and embed data
    embedder = Word2Vec("skip_s300.txt")

    data_loader = dict()
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(DATA_PATH, f"{split}_sa.csv"))
        X = df['review'].to_numpy()
        X = embedder.get_idx(X, max_seq_len)

        Y = df['rating'].apply(lambda x: 0 if x == -1 else 1)
        n_class = Y.unique().shape[0]
        Y = Y.to_numpy()

        print(X.shape, Y.shape)
        
        data_loader[split] = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=batch_size, shuffle=True)
        
    ## Train model
    embedding_matrix = embedder.embedding_matrix
    model = LSTM(embedding_matrix, n_class, max_seq_len, torch.device('cuda:0'))
    model.do_training(
        n_epochs,
        batch_size,
        data_loader["train"],
        data_loader["dev"],
        print_every,
        writer,
    )

    ## Test model
    Y = embedder.get_idx()
    labels, preds, loss = model.do_test(batch_size, data_loader["test"], writer)

    writer.add_scalar("Loss/test", loss.item())
    print(f"Test loss: {loss}")

    print(classification_report(labels, preds))

    writer.add_figure(ConfusionMatrixDisplay(labels, preds), 'Confusion matrix')

    writer.close()


if __name__ == "__main__":
    main()
