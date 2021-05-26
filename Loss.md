# **神经网络Loss相关知识**

## **triplet loss**
[triplet loss link](http://bindog.github.io/blog/2019/10/23/why-triplet-loss-works/)
### **triplet loss简介**
<font color="black">深度学习领域有一块非常重要的方向称之为metric learning，其中一个具有代表性的方法就是triplet loss，triplet loss的基本思想很清晰，就是让同一类别样本的feature embedding尽可能靠近，而不同类别样本的feature embedding尽可能远离，其中样本的feature embedding是通过同一个深度神经网络抽取得到的。</font>

在triplet loss中，我们会选取一个三元组，首先从训练集中选取一个样本作为Anchor，然后再随机选取一个与Anchor属于同一类别的样本作为Positive，最后再从其他类别随机选取一个作为Negative，这里将样本的feature embedding记为x，那么一个基本的三元组triplet loss如下：
$$l_{tri}=\max (\Vert x_a - x_p \Vert - \Vert x_a - x_n \Vert + \alpha, 0)$$
其中α为margin；上式要求Negative到Anchor的距离至少要比Positive到Anchor的距离大α，显然α越大不同类别之间的可区分性就越强，相应的训练难度也越大。当然也可以把α设为0，这样条件就放的比较宽松了，但是triplet loss的优势也就很难体现出来了。

<img src="https://i.loli.net/2019/10/23/Yr3I9ayivw5cXou.png" width=60%>

在triplet loss基础上，又衍生出了其他许多改进和变体，例如一个比较有效的方法叫hard mining，在三元组选择过程中加入一些特定的策略，尽量选择一些距离Anchor较远的Positive和距离Anchor较近的Negative（也就是最不像的同类样本、最像的不同类样本）……此类方法还有许多，就不一一列举了。

<font color = 'black'>然而triplet loss虽然有效，但是其常为人诟病的缺点也很明显：训练过程不稳定，收敛慢，需要极大的耐心去调参……所以在很多情况下，我们不会单独使用triplet loss，而是将其与softmax loss等方法相结合使用，以稳定训练过程。</font>

### **理论分析**
我们回到triplet loss当中三元组的基本形式，首先约定一些符号（与上面提到的那篇论文保持一致），假定训练集样本总量为N，定义同类别样本集合 
$$S = \{(i,j) \mid y_i = y_j \}_{i,j\in \{1,\cdots N \}}$$
，那么最基本的triplet loss表达式形式如下（为了简化问题，我们暂时忽略了margin这一项）

$$L_t(T,S)=\sum\limits_{(i,j)\in S,(i,k)\notin S,i,j,k\in\{1,\cdots,N\}}l_t(x_i,x_j,x_k) \tag{1}$$
$$l_t(x_i,x_j,x_k)=\Vert x_i-x_j \Vert - \Vert x_i - x_k \Vert \tag{2}$$
直接对上面两个式子做变换是比较困难的，如果假定每一个类别都有自己的一个类中心cm，当然这个类中心和所有的样本处于同一个embedding空间， $\mathcal{C}=\{c_m\}_{m=1}^C$，其中 $c_m\in \mathbb{R}^D$。

通过引入这样一个辅助的类中心点，我们可以利用三角不等式写出如下式子：
$$\Vert x_i-x_j \Vert \leq \Vert x_i-c_{y_i} \Vert + \Vert x_j-c_{y_i} \Vert \tag{3}$$
$$\Vert x_i - x_k \Vert \geq \Vert x_i-c_{y_k} \Vert - \Vert x_k-c_{y_k} \Vert \tag{4}$$

根据上面几个式子，我们可以写出 $l_t(x_i,x_j,x_k)$,的一个上界 $l_d(x_i,x_j,x_k)$ ，如下:

$$l_d(x_i,x_j,x_k)=\Vert x_i-c_{y_i} \Vert - \Vert x_i-c_{y_k} \Vert + \Vert x_j-c_{y_i} \Vert + \Vert x_k-c_{y_k} \Vert \tag{5}$$

当然这只是其中一个三元组的loss，还不太容易观察出规律，如果考虑整个训练集上所有可能的三元组，累加得到整体的triplet loss就可以看出一些有意思的东西了。那么符合条件的三元组有多少个呢？为了简化分析流程，我们要做一个合理的假定，那就是训练集当中每个类别的样本数量是相等的（不相等的话可以通过采样的方式使之相等），通过简单的排列组合知识可以得出三元组的个数为 $A_C^2 \cdot A_{\frac{N}{C}}^2 \cdot C_{\frac{N}{C}}^1$；然后我们仔细观察式(5)可以发现，它实际上是由两部分构成的，一部分是样本减去其对应的类中心，另一部分是样本减去其他类别的类中心。所以稍微整合一下即可得到整体的triplet loss，如下所示：

$$L_d(T,S)=G\sum\limits_{i=1}^N(\Vert x_i-c_{y_i} \Vert - \frac{1}{3(C-1)}\sum\limits_{m=1,m\neq y_i}^C \Vert x_i-c_m \Vert) \tag{6}$$
其中 
$$G=3(C-1)(\frac{N}{C}-1)\frac{N}{C}$$