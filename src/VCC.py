#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[16]:


import time
import torch
from torch import nn, optim
import torchvision
import sys
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)


# In[7]:
def load_data_fashion_mnist(batch_size,resize = None, root='~/Datasets/FashionMNIST'):
     """Download the fashion mnist dataset and then load into memory."""
     trans = []
     if resize:
         trans.append(torchvision.transforms.Resize(size=resize))
     trans.append(torchvision.transforms.ToTensor())
     transform = torchvision.transforms.Compose(trans)
     mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
     mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
     if sys.platform.startswith('win'):
         num_workers = 0  # 0表示不用额外的进程来加速读取数据
     else:
         num_workers = 4
     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
     print('finish read data')
     return train_iter, test_iter



class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)



def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net


# In[23]:

'''
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

net = vgg(conv_arch, fc_features, fc_hidden_units)

'''
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意
ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

# In[24]:

'''
net = vgg(small_conv_arch,fc_features,fc_hidden_units)
X = torch.rand(1,1,224,224)
for name,blk in net.named_children():
    X = blk(X)
    print(name,'output shape :',X.shape)
'''

# In[25]:


batch_size = 32
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size,resize = 224)


# In[28]:


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

                



lr = 0.001
num_epochs = 500

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    


train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

