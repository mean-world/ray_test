
def train_pytorch_model():
    import logging
    import os
    from torchvision import transforms, datasets
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist

    import yfinance as yf
    import numpy as np
    from torch.utils.data import Dataset
    import pandas as pd
    import torch.nn as nn
    from sklearn import preprocessing

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG,
    )

    # Create PyTorch attention LSTM Model.
    
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()

            self.W_b = nn.Linear(hidden_size, hidden_size)
            nn.init.xavier_normal_(self.W_b.weight)

            self.v_b = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(1, hidden_size)))
            self.W_e = nn.Linear(hidden_size*2, hidden_size)

        def forward(self, input, ht_ct):
            a = self.W_e(ht_ct)
            b = self.W_b(input)
            c = a + b
            j = torch.tanh(c) @ self.v_b.T
            return F.softmax(j, dim=0)


    class LSTMLayer_bacth_version(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers):
            super(LSTMLayer_bacth_version, self).__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)

        def forward(self, x):
            h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()

            out, _ = self.lstm(x, (h0.detach(), c0.detach()))
            return out

    
    class att(nn.Module):
        def __init__(self, hidden_size):
            super(att, self).__init__()
            self.attention_layer = AttentionLayer(hidden_size)  
            self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        def forward(self, x):
            output = torch.zeros(x.size(0), x.size(1), x.size(2))
            h_t = torch.zeros(x.size(1), x.size(2)).requires_grad_()
            c_t = torch.zeros(x.size(1), x.size(2)).requires_grad_()

            #batch
            for i in range(x.size(0)):
                #time window
                for j in range(x.size(1)):
                    #feature correlation
                    h_t_tw = torch.zeros(1, x.size(2)).requires_grad_()
                    c_t_tw = torch.zeros(1, x.size(2)).requires_grad_()

                    h_c_concat = torch.cat((h_t_tw, c_t_tw), dim=1)
                    weight = self.attention_layer(x[i, j, :], h_c_concat)*x[i, j, :]
                    h_t_tw, c_t_tw = self.lstm_cell(weight.view(1, x.size(2)), (h_t_tw, c_t_tw))
                    output[i, j, :] = h_t_tw
                
                #temporal correlation
                h_c_concat = torch.cat((h_t, c_t), dim=1)
                weight = self.attention_layer(x[i, :, :], h_c_concat)*x[i, :, :]
                h_t, c_t = self.lstm_cell(weight, (h_t, c_t))
                output[i, :, :] = h_t
                    
            return output
            
    class LSTM_model(nn.Module):
        def __init__(self, hidden_size):
            super(LSTM_model, self).__init__()
            self.lstm = LSTMLayer_bacth_version(6, hidden_size, 1)
            self.lstm_2 = LSTMLayer_bacth_version(hidden_size, 6, 1)
            self.lr = nn.Linear(6, 6)
            self.relu = torch.nn.ReLU()
            self.att_layer = att(hidden_size)
        def forward(self, x):
            out = self.lstm(x)
            out = self.relu(out)
            out = self.att_layer(out)
            out = self.lstm_2(out)
            out = self.relu(out)
            out = self.lr(out)
            out = torch.sigmoid(out)
            return out[:, -1, :]

    
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    
    model = LSTM_model(16)
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD_SIZE)
    Distributor = nn.parallel.DistributedDataParallel
    model = Distributor(model)

    # Split batch size for each worker.
    batch_size = int(128 / WORLD_SIZE)

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

    train_set, train_label, test_set, test_label = split_data(data, 6, 0.8)

    class data_set(Dataset):
        def __init__(self, data, transform=None, target_transform=None):
            self.transform = transform
            self.target_transform = target_transform
            self.train, self.label = data

        def __len__(self):
            return self.train.size(0)

        def __getitem__(self, index):
            return self.train[index, :, :], self.label[index, :]
        
    train_dataset = data_set((train_set, train_label))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # Start Training.
    logging.info(f"Start training for RANK: {RANK}. WORLD_SIZE: {WORLD_SIZE}")
    for epoch in range(10):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mse_loss = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = mse_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    import mlflow
    import socket
    if socket.gethostname() == "train-pytorch-master-0":
        mlflow.set_tracking_uri("https://mlflow-server:5000")
        with mlflow.start_run() as run:
            mlflow.pytorch.save_model(pytorch_model = model)


from kubeflow.training import TrainingClient
from kubeflow.training import constants

pytorchjob_name = "train-pytorch"
training_client = TrainingClient(job_kind=constants.PYTORCHJOB_KIND)
training_client.create_job(name=pytorchjob_name, train_func=train_pytorch_model, num_worker_replicas=3, packages_to_install = ["yfinance", "scikit-learn", "mlflow"],)
# training_client.delete_job(pytorchjob_name)