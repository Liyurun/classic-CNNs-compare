
import time
import torch
from torch import nn,optim
import torchvision.transforms as transforms
import torchvision
import sys

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
         num_workers = 0
     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
     print('finish read data')
     return train_iter, test_iter



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


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    test_acc,re_train_loss,re_train_acc,re_test_acc,re_time = [],[],[],[],[]
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
        re_train_loss.append(train_l_sum / batch_count)
        re_train_acc.append(train_acc_sum / n)
        re_test_acc.append(test_acc)
        re_time.append(time.time() - start)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return (re_train_loss,re_train_acc,re_test_acc,re_time)

