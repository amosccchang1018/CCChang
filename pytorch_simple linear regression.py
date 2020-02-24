# -*- coding: utf-8 -*-
"""
create simple regression to fit data

Created on Mon Feb 24 09:00:33 2020
@author: Chi
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------

# create sample data first
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
y = x.pow(2) + 0.2*torch.rand(x.size())                 



class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.predict(x)             
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     
print(net)  #

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # use SGD
loss_func = torch.nn.MSELoss()  

plt.ion()   # plotting

for t in range(200):
    prediction = net(x)     

    loss = loss_func(prediction, y)     

    optimizer.zero_grad()
    loss.backward()        
    optimizer.step()       

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 18, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()