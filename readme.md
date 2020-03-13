# classic-CNNs-compare
对比了几种经典的卷积神经网络的算法，方便自己写的网络与经典的几种算法进行对比，包括简单卷积神经网络（LeNet）^[1]^ 、深度卷积神经网络（AlexNet）^[2]^ 、包含重复元素的卷积神经网络（VGG）^[3]^ 、嵌入网络的卷积神经网络（NiN）^[4]^ 、并行连接的卷积神经网络（GoogLeNet）^[5]^ 、残差卷积神经网络（ResNet）^[6]^ 。

程序主要包括3个部分：
**compare.py** 调用各类CNN模型，定义数据集，优化算法
**collect_net.py** 定义CNN模型，以及需要的模块
**utils**  定义辅助函数，包括准确度计算，训练模式。

模型的参数都按照相同的参数进行计算，需要特殊结构的，如GoogLeNet中的Inception和ResNet中的Residual模块，具体参数设置在collect_net.py。

##训练误差

![](https://github.com/Liyurun/classic-CNNs-compare/blob/master/results/train_acc.png=50x50)



##测试误差
![](https://github.com/Liyurun/classic-CNNs-compare/blob/master/results/test_acc.png)


##损失函数
![](https://github.com/Liyurun/classic-CNNs-compare/blob/master/results/loss.png)



[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
[3] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
[4] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
[5] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 4, p. 12).
[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings in deep residual networks. In European Conference on Computer Vision (pp. 630-645). Springer, Cham.


