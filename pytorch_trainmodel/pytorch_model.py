import torch
import torch.nn as nn
class classify(nn.Module):
  def __init__(self, phobert, number_of_category):
    super(classify,self).__init__()
    self.phobert=phobert
    self.relu=nn.ReLU()
    self.dropout=nn.Dropout(0.1)
    self.first_function=nn.Linear(768, 512)
    self.second_function=nn.Linear(512, 32)
    self.third_function=nn.Linear(32,number_of_category)
    self.softmax=nn.LogSoftmax(dim=1)

  def forward(self, input):
    x=self.phobert(input)
    x=self.first_function(x[1])
    x=self.relu(x)
    x=self.dropout(x)
    x=self.second_function(x)
    x=self.relu(x)
    x=self.third_function(x)
    x=self.softmax(x)
    return x