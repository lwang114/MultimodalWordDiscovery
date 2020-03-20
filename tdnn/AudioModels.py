from TDNN.tdnn import *
import torch.nn as nn
import torch.nn.functional as F

class TDNN3(nn.Module):
  def __init__(self, n_class, embedding_dim=128):
    super(TDNN3, self).__init__()
    self.embedding_dim = embedding_dim
    self.batchnorm1 = nn.BatchNorm2d(1)
    # self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 3), stride=(1, 1), padding=(0, 1))
    # XXX
    '''
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(40, 3), stride=(1, 1), padding=(0, 1))
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    self.conv3 = nn.Conv2d(64, embedding_dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)) 
    # XXX
    
    self.avgpool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0))
    
    self.fc = nn.Linear(embedding_dim, n_class) 
    '''
    '''
    self.td1 = TDNN(input_dim=input_dim, 
                    output_dim=10,
                    context_size=1)
    self.td2 = TDNN(input_dim=input_dim, 
                    output_dim=5,
                    context_size=2)
    '''
    self.fc1 = nn.Linear(600, 1000)
    self.fc2 = nn.Linear(1000, n_class)

  def forward(self, x, save_features=False):
    '''
    if x.dim() == 3:
      x = x.unsqueeze(1)
    '''
    x = x.view(x.size(0), -1)
    '''
    x = self.batchnorm1(x)
    x = F.relu(self.conv1(x))
    # XXX
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    '''
    '''
    x = self.avgpool(x)
    embed = x.squeeze()'''
    '''
    embed = torch.mean(x, -1).squeeze()
    out = self.fc(embed)
    '''
    embed = F.relu(self.fc1(x))
    out = self.fc2(embed)

    if save_features:
      return embed, out
    else:
      return out

class BLSTM2(nn.Module):
  def __init__(self, n_class, embedding_dim=100, n_layers=1):
    super(BLSTM2, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    # self.i2h = nn.Linear(40 + embedding_dim, embedding_dim)
    # self.i2o = nn.Linear(40 + embedding_dim, n_class) 
    self.rnn = nn.LSTM(input_size=40, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x.unsqueeze(0)

    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    embed, _ = self.rnn(x, (h0, c0))
    outputs = []
    for b in range(B):
      # out = self.softmax(self.fc(embed[b]))
      out = self.fc(embed[b])
      outputs.append(out)

    if save_features:
      return embed, torch.stack(outputs, dim=1)
    else:
      return torch.stack(outputs, dim=1)
