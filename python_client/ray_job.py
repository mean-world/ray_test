import yfinance as yf
from typing import Dict
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
import tempfile
import os


import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import model as model_fw
lstm_model = model_fw.LSTM_model(16)

# ray.init(_node_ip_address='172.17.0.3')

#custom dataset class
class data_set(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train, self.label = data

    def __len__(self):
        return self.train.size(0)

    def __getitem__(self, index):
        return self.train[index, :, :], self.label[index, :]



# Get stock DataSet.
#web crawler
#'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
start_date = '2009-01-01'
end_date = '2023-11-08'
ticker = 'GOOGL'
data = yf.download(ticker, start_date, end_date)


#normalize data
def normalize(data):
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    return data
data = normalize(data)

#data preprocess
def split_data(stock, window_size, rate):
    data_raw = stock.to_numpy()
    data = []

    for i in range(len(data_raw) - window_size):
        data.append(data_raw[i: i + window_size])

    data = np.array(data)
    test_set_size = int(np.floor(rate * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size,:-1]
    y_train = data[:train_set_size,-1]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]

# train_set, train_label, test_set, test_label = split_data(data, 6, 0.8)

def get_dataloaders(batch_size):
    train_set, train_label, test_set, test_label = split_data(data, 6, 0.8)

    train_dataset = data_set((train_set, train_label))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = data_set((test_set, test_label))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Get dataloaders inside the worker training function
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    # =======================================================================
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    model = lstm_model

    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    # ============================================================
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Model training loop
    for epoch in range(epochs):
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()

        test_loss /= len(test_dataloader)
        

        # [3] Report metrics to Ray Train
        # ===============================
        torch.save(model.module.state_dict(),"/root/model.pt")
        ray.train.report(metrics={"loss": test_loss}, checkpoint=ray.train.Checkpoint.from_directory("/root"))
        


def train_fashion_mnist(num_workers=2, use_gpu=False):
    global_batch_size = 32

    train_config = {
        "lr": 1e-3,
        "epochs": 2,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")
    print(result)





train_fashion_mnist(num_workers=1, use_gpu=False)

