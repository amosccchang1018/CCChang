# -*- coding: utf-8 -*-
"""
create simple model to classify the data (2 or more group)

Created on Mon Feb 24 09:06:07 2020
@author: Chi
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#sample data 

n_data = torch.ones(120, 2)
a = torch.normal(2*n_data, 1)      
b = torch.zeros(120)               
a1 = torch.normal(-2*n_data, 1)     
b1 = torch.ones(120)               
x = torch.cat((a, a1), 0).type(torch.FloatTensor)  
y = torch.cat((b, b1), ).type(torch.LongTensor)    


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.out = torch.nn.Linear(n_hidden, n_output)   

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.out(x)
        return x

# network
net = Net(n_feature=2, n_hidden=10, n_output=2)     
print(net)  

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  


#visualize
plt.ion()  

for t in range(100):
    out = net(x)                 
    loss = loss_func(out, y)    

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()       

    if t % 2 == 0:
        
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
