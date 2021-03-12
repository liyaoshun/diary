# <div align = center>**论文阅读日志** </div>

---
## **Autonomous driving 自动驾驶**
### **《MapFusion: A General Framework for 3D Object Detection with HDMaps》** [Paper](https://arxiv.org/pdf/2103.05929.pdf)

提出了一个简单且有效的框架--MapFusion. MapFusion将地图信息集成到现代三维物体探测器管道中.

---
## **语义分割相关**
### **《A Good Box is not a Guarantee of a Good Mask》**

    在这项工作中，主要考虑了LVIS数据集的两个特征：长尾分布和高质量实例分割mask。
    
    采用两阶段的训练流程。在第一阶段(训练)，结合了EQL和自训练来学习泛化表示。在第二阶段(微调)，利用Balanced GroupSoftmax来促进分类器的改进，并提出一种新颖的proposal 分配策略和一种针对mask head的新的平衡mask损失，以获取更精确的mask预测。

    在训练阶段使用了EQL损失函数、RFS重采样、DA、Self-training等策略。在微调阶段会freeze backbone参数，然后使用balanced group softmax 进行类别平衡，同时在微调阶段更加的重视mask的结果。
    
    使用到的Tricks：
    1. DA（Data Augmentation）：Mosaic, rotate, scale jitter
    2. EQL（Equalization Loss）
    3. RFS（Repeat Factor Sampling）
    4. HTC（Hybrid Task Cascade）
    5. Self-training
    6. ...
[Balanced Group SoftMax Github](https://github.com/FishYuLi/BalancedGroupSoftmax)

**实验指标如下表：**
<div align=center>
<img src="Paper/LVIS_0.png">
<img src="Paper/LVIS_1.png">
</div>
---

### **《OCRNet》 "基于物体区域的上下文信息进行语义分割"**
    微软亚洲研究院提出的 OCR 方法的主要思想是显式地把像素分类问题转化成物体区域分类问题，这与语义分割问题的原始定义是一致的，即每一个像素的类别就是该像素属于的物体的类别，换言之，与 PSPNet 和 DeepLabv3 的上下文信息最主要的不同就在于 OCR 方法显式地增强了物体信息。

    OCR 方法的实现主要包括3个阶段：
    1. 根据网络中间层的特征表示估测一个粗略的语义分割结果作为 OCR 方法的一个输入 ，即软物体区域（Soft Object Regions），
    2. 根据粗略的语义分割结果和网络最深层的特征表示计算出 K 组向量，即物体区域表示（Object Region Representations），其中每一个向量对应一个语义类别的特征表示，
    3. 计算网络最深层输出的像素特征表示（Pixel Representations）与计算得到的物体区域特征表示（Object Region Representation）之间的关系矩阵，然后根据每个像素和物体区域特征表示在关系矩阵中的数值把物体区域特征加权求和，得到最后的物体上下文特征表示 OCR (Object Contextual Representation) 。
    当把物体上下文特征表示 OCR 与网络最深层输入的特征表示拼接之后作为上下文信息增强的特征表示（Augmented Representation），可以基于增强后的特征表示预测每个像素的语义类别。
    
    综上，OCR 可计算一组物体区域的特征表达，然后根据物体区域特征表示与像素特征表示之间的相似度将这些物体区域特征表示传播给每一个像素。

<font color=red>Eg: 在进行15类别分割的时候，通过pink框计算得到 N * 15 * H * W的soft object regions.然后将其和backbone输出的特征进行计算得到物体的区域表示特征Object Region Representations（N * channels * 15 * 1）,此矩阵表示每一个物体类别由channels维度描述子描述.然后使用此描述子和backbone的输出进行相似度计算，然后根据每个像素和物体区域特征表示在关系矩阵中的数值把物体区域特征加权求和，得到最后的物体上下文特征表示。最后拼接主干网络的输出特征和和最终的上下文特征作为增强后的特征。</font>

下图为OCR的pipeline.
<div align=center>
    <img src = "Paper/ocr_pipeline.png" />
</div>
通过实验对比，OCR 方法提出的物体上下文信息的目的在于显式地增强物体信息，通过计算一组物体的区域特征表达，根据物体区域特征表示与像素特征表示之间的相似度将这些物体区域特征表示传播给每一个像素。在街景分割任务中，OCR 方法也比 PSPNet 的 PPM  和 DeepLabv3 的 ASPP更加高效也更加准确。

---
### **《Strip Pooling: Rethinking Spatial Pooling for Scene Parsing》**
作者开源代码： [code](https://github.com/Andrew-Qibin/SPNet)

1. 一般的空间均值池化能捕获到环境的上下文信息，但是由于使用的池化kernel的shape是正方形的，所以在环境中的一些非类似正方形物体上进行上下文信息捕获的时候会得到一些噪音，使得其对物体的描述产生影响。(提高CNN远程关系建模的方法有：扩张卷积、全局/金字塔池化)。在原文中的Figure.1能很好的体现这种影响，下图问copy-Figure.1.
<div align=center>
    <img src = "Paper/strip_pooling_0.jpg" />
</div>
<!-- ![Images text](/Paper/strip_pooling_0.jpg) -->

**优点**:首先，它沿着一个空间维度部署一个长条状的池化核形状，因此能够捕获孤立区域的长距离关系，如图1a和1c的第一行所示部分所示。其次，在其他空间维度上保持较窄的内核形状，便于捕获局部上下文，防止不相关区域干扰标签预测。集成这种长而窄的池内核使语义分割网络能够同时聚合全局和本地上下文。这与传统的从固定的正方形区域收集上下文的池化有本质的不同。二、方法基于条纹池化的想法，作者提出了两种即插即用的池化模块 — Strip Pooling Module (SPM) 和Mixed Pooling module (MPM)。

2. SPM，实际上和普通池化方法没有区别，就是把池化核（长条形区域）所对应的特征图上位置的像素值求平均
![Images text](/Paper/strip_pooling_spm.jpg)
再来看一下整个模块，输入一个特征图，这里实际上为C×H×W，为了方便讲解图中只画了一个通道。我们也以一个通道为例。C个通道的特征图输入处理原理与这里一模一样。输入的特征图经过水平和数值条纹池化后变为H×1和1×W。随后经过卷积核为3的1D卷积与expand后再对应相同位置求和得到H×W的特征图。之后通过1×1的卷积与sigmoid处理后与原输入图对应像素相乘得到了输出结果。//
**在上面的过程中，输出张量中的每个位置都与输入张量中的各种位置建立了关系。例如，在上图中，输出张量中以黑框为界的正方形与所有与它具有相同水平或垂直坐标的位置相连(被红色和紫色边框包围)。因此，通过多次重复上述聚合过程，可以在整个场景中构建长期依赖关系。（我觉得和CCNet 异曲同工之妙）此外，得益于elementwise乘法操作，该SPM也可以被视为一种注意机制。**//
<font color=red>SPM可以直接应用于任何预先训练的骨干网络，而无需从无到有地进行训练。</font>
//
与全局平均池化相比，条纹池化考虑的是较长但较窄的范围，而不是整个特征图，避免了在相距较远的位置之间建立不必要的连接。与需要大量计算来建立每对位置之间关系的基于注意力的模块（no-local ）相比，SPM是轻量级的，可以很容易地嵌入到任何构建块中，从而提高捕获远程空间依赖关系和利用通道间依赖项的能力。

3. MPM，进一步在高语义级别上建模长期依赖关系。它通过利用具有不同内核形状的池化操作来探测具有复杂场景的图像，从而收集有用的上下文信息。之前的研究结果表明，金字塔池模型(pyramid pooling module, PPM)是增强语义分割网络的有效方法。然而，PPM严重依赖于标准的池化操作(尽管不同的池内核位于不同的金字塔级别)。考虑到标准池化和条纹池化的优点，作者改进了PPM，提出了混合池模块(MPM)，它侧重于通过各种池化操作聚合不同类型的上下文信息，以使特征表示更有辨别力。
<!-- ![Images text](/Paper/strip_pooling_mpm.png) -->
<div align=center>
<img src ="Paper/strip_pooling_mpm.png"/>

</div>
1. 实验结果可视化
<!-- ![Images text](/Paper/strip_pooling_result.png) -->
<div align=center>
<img src ="Paper/strip_pooling_result.png"/>
<img src ="Paper/strip_pooling_result_table.png"/>
</div>
<!-- [Images text](/Paper/strip_pooling_result_table.png) -->

## **致谢**
https://zhuanlan.zhihu.com/p/122571198

---
   