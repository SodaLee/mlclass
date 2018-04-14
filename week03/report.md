# 机器学习概论#

1. 算法设计

   本次作业在cifar10数据集上研究了图像分类算法

   使用的工具为tensorflow，所使用的网络结构为Resnet-18

2. 网络结构

   Resnet-18的网络结构和普通卷积神经网络略有不同，加入了残差的概念，在输出函数与0相比更接近输入的情况下，残差可能比0函数更容易学习。同时对于更深的网络而言，梯度更不容易消失，因此，我们可能期待resnet在更深的网络层数情况下对任务有更好的表现。

   ```python
   网络参数
   5x5conv
   2x2pool
   Block1: 3x3conv * 2, out = input + conv_2
   Block2:
   2x2pool + channel * 2
   Block3:
   Block4:
   2x2pool + channel * 2
   Block5:
   Block6:
   2x2pool + channel * 2
   Block7:
   Block8:
   Global pool
   Fc - > 10
   ```

3. 结果

   训练20 epoch的情况下在测试集上达到了81%的正确率，网络还未完全收敛正确率仍在缓慢的提升，根据所了解到的资料，resnet的训练一般都需要约100个epoch才能够收敛，预期正确率可能会大于90%

   30epoch:82.5%

   35epoch:83.2%