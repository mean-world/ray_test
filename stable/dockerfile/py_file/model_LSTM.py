import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        hidden_size = 16
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


