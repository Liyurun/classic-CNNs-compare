#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import time
import torch
from torch import nn,optim
import torchvision.transforms as transforms
import torchvision
import sys


# In[32]:



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    def forward(self,img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output


# In[33]:


net = LeNet()
net


# In[34]:


def load_data_fashion_mnist(batch_size, root='~/Datasets/FashionMNIST'):
     """Download the fashion mnist dataset and then load into memory."""
     transform = transforms.ToTensor()
     mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
     mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
     if sys.platform.startswith('win'):
         num_workers = 0  # 0表示不用额外的进程来加速读取数据
     else:
         num_workers = 4
     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

     return train_iter, test_iter


# In[35]:


batch_size = 256
train_iter,test_iter = load_data_fashion_mnist(batch_size=batch_size)


# In[ ]:

def evaluate_accuracy(data_iter,net,device = None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# In[ ]:


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
        net = net.to(device)
        print("train on ",device)
        loss = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            train_l_sum,train_acc_sum,n,batch_count,start = 0.0,0.0,0,0,time.time()
            for X,y in train_iter:
                X = X.to(device)
                y = y.to(device)
                print(X.shape)

                for name, blk in net.named_children(): 
                    print(blk)
                    X = blk(X)
                    print(name, 'output shape: ', X.shape)
                
                
                y_hat = net(X)
                l = loss(y_hat,y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = evaluate_accuracy(test_iter,net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

                


# In[ ]:


lr,num_epochs = 0.001,500
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

