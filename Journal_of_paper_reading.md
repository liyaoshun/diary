# **论文阅读日志**

## **语义分割相关**
### **《Strip Pooling: Rethinking Spatial Pooling for Scene Parsing》**
作者开源代码： [code](https://github.com/Andrew-Qibin/SPNet)

1. 一般的空间均值池化能捕获到环境的上下文信息，但是由于使用的池化kernel的shape是正方形的，所以在环境中的一些非类似正方形物体上进行上下文信息捕获的时候会得到一些噪音，使得其对物体的描述产生影响。(提高CNN远程关系建模的方法有：扩张卷积、全局/金字塔池化)。在原文中的Figure.1能很好的体现这种影响，下图问copy-Figure.1.
    <div align=center>
        <img src = "Paper/strip_pooling_0.jpg" />
    <div/>
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
   