# **异常检测相关知识**

## **简介**
[LINK](https://mp.weixin.qq.com/s?__biz=MzIxOTczOTM4NA==&mid=2247509039&idx=2&sn=a683d3aa9490d2e3195e7be5d756816c&chksm=97d441b8a0a3c8aec2360267522081f3db21fda276a7ec9881ed410829a5e9c89c519f8d4ed2&scene=0&xtrack=1&key=56a3b1411e85bb000bbd80aa6e0282fee39d8740d157a54131eba4101fb2c281a22da0d37f169044d5ee9cf1f60044d7862da7dffa5c4a250f00346aa67ed4aa21f7933c57cbeec6f5714e0cc5205e2e8cdede890f22ef3d4efa6b641b64460ddb035ede566204bb5225c93d46099db5487bba4f04eff6d16eac783eb885066b&ascene=1&uin=MTc5OTk4OTk0Mg%3D%3D&devicetype=Windows+XP&version=62060841&lang=en&exportkey=A%2BhjME%2F0XvxjW1Z%2B5l2lDl0%3D&pass_ticket=OCJlm83MoN5T6b0IyDIDr%2B2ilg9kWGap6J1HDcyYS85zh%2FxxYOcPZtrjljf7X3qP&wx_header=0)
如果将每个样本看作空间中的一个点的话，异常（anomaly）简单地来说就是离群点，或者说是“非主流”的点，即离大多数样本点都比较远。这里隐藏的意思是，异常通常是少数。
下图很形象地展示了什么是异常。其中，黑色的点为正常的样本点，红色的点为异常点样本点。（更多的介绍可以参考Kiwi：异常检测概述（一）：An Overview of Anomaly Detection Part I [LINK](ttps://zhuanlan.zhihu.com/p/50384515)）

**难点**：目前实际的异常检测遇到的一个很大的困难，是在实际的场景中（例如工业流水线等），异常样本往往很难获得，甚至很多时候没有异常样本。这就迫使我们采用semi-supervised或者unsupervised的方法。

## **图像空间上的异常检测**
图像空间上的异常检测一般采用的是下图的Auto-encoder结构。主要的思想是，如果我们只在正常样本的训练集上训练我们的Auto-encoder，则我们重构出来的图像会比较趋近于正常样本。利用这一个假设/性质，在推理阶段，即使输入的图像是异常样本，我们重构出来的图像也会像正常样本。所以我们只需对比原图和重构后的图，就可以判断输入的图像是否为异常，甚至可以定位到异常区域。
