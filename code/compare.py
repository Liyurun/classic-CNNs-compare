import time
import torch
from torch import nn,optim
import torchvision.transforms as transforms
import torchvision
import sys
import collect_net
import utils as ut
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import caltime as ct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
train_iter, test_iter = ut.load_data_fashion_mnist(batch_size,resize=224)
net_dict={}
iter_num={}
#vgg
VggNet = collect_net.VggNet()
net_dict['VggNet'] = VggNet
iter_num['VggNet'] = 50

#nin
NiNNet = collect_net.NiN()
net_dict['NinNet'] = NiNNet
iter_num['NinNet'] = 150
#LeNet
LeNet = collect_net.LeNet()
net_dict['LeNet'] =  LeNet
iter_num['LeNet'] = 250
#AlexNet
AlexNet = collect_net.AlexNet()
net_dict['AlexNet'] = AlexNet
iter_num['AlexNet'] = 150
#vgg


#Res
ResNet = collect_net.ResNet() 
net_dict['ResNet'] = ResNet
iter_num['ResNet'] = 100

#GoogLeNet
GoogLeNet = collect_net.GoogLeNet()
net_dict['GoogLeNet'] = GoogLeNet
iter_num['GoogLeNet'] = 100


lr,num_epochs = 0.001,50
record={}
for net_name, net_value in net_dict.items():
    print('start '+net_name)    
    params = ()
    net = net_dict[net_name].cuda()
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
    num_epochs = iter_num[net_name]
    params = ut.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    record[net_name] = params


filename = 'a23.pkl'
dill.dump_session(filename)

#re_train_loss,re_train_acc,re_test_acc,re_time

def accum(ls):
    sum  = 0
    ls1 = []
    for i in ls:
        sum += i
        ls1.append(sum)
        
    return ls1

train_loss,train_acc,test_acc,time_used = [],[],[],[]

for i,(key,value) in enumerate(record.items()):
    train_loss.append(value[0])
    train_acc.append(value[1])
    test_acc.append(value[2])
    time_used.append(value[3])
train_loss

for i,time in enumerate(time_used):
    print(time)
    time_used[i] = accum(time)
    print(time_used)

plt.figure()
for i,(a,b,c) in enumerate(zip(train_loss,time_used,record.keys())):
    plt.plot(b,a,label = c)
    plt.legend()

plt.xlabel('time used')
plt.ylabel('loss')
plt.savefig("loss.png")

plt.figure()
for i,(a,b,c) in enumerate(zip(train_acc,time_used,record.keys())):
    plt.plot(b,a,label = c)
    plt.legend()

plt.xlabel('time used')
plt.ylabel('train_acc')
plt.savefig("train_acc.png")

plt.figure()
for i,(a,b,c) in enumerate(zip(test_acc,time_used,record.keys())):
    plt.plot(b,a,label = c)
    plt.legend()

plt.xlabel('time used')
plt.ylabel('test_acc')
plt.savefig("test_acc.png")









