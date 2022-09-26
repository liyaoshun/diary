**GPU 计算相关知识，主要是 GPU 线层坐标计算相关问题**

**thread 索引的计算方式**

```
用公式表示：最终的线程Id = blockId * blockSize + threadId
1. blockId ：当前 block 在 grid 中的坐标（可能是1维到3维）
2. blockSize ：block 的大小，描述其中含有多少个 thread
3. threadId ：当前 thread 在 block 中的坐标（同样从1维到3维）

关键点：
1. grid中含有若干个blocks，其中 blocks 的数量由 gridDim.x/y/z 来描述。
某个 block 在此 grid 中的坐标由 blockIdx.x/y/z 描述。
2. blocks 中含有若干个 threads，其中 threads 的数量由 blockDim.x/y/z 来描述。
某个 thread 在此 block 中的坐标由 threadIdx.x/y/z 描述。
```

**1D grid, 1D block 类型的 id 计算**

```
blockSize = blockDim.x
blockId = blockIdx.x
threadId = threadIdx.x
Id = blockIdx.x * blockDim.x + threadIdx.x
```

**3D grid, 1D block 类型的 id 计算**

```
blockSize = blockDim.x（一维 block 的大小）
blockId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x
threadId = threadIdx.x
Id = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x
```

**1D grid, 2D block 类型的 id 计算**

```
blockSize = blockDim.x * blockDim.y（二维 block 的大小）
blockId = blockIdx.x（一维 grid 中 block id）
threadId = blockDim.x * threadIdx.y + threadIdx.x
Id = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x
```

**3D grid, 3D block 类型的 id 计算**

```
blockSize = blockDim.x * blockDim.y * blockDim.z（三维 block 的大小）
blockId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x
threadId = blockDim.x * blockDim.y * threadIdx. z + blockDim.x * threadIdx.y + threadIdx.x
Id = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) + blockDim.x * blockDim.y * threadIdx. z + blockDim.x * threadIdx.y + threadIdx.x
```

**一些解释**

```
CUDA中

grid、block、thread只要在最大值范围内，可以随意设置，最后系统会根据内部算法重新分配。因此grid、block、thread的设置其实只以你的“目的”有关。也就是说这几个值的索引如何方便如何来，要处理的矩阵就用二维，要处理的向量就用一维，要处理的张量就用三维

在设置上，变量是遵循向下设置的， 即

dim3 grid(1, 1, 1)表示的是一个grid由一个block组成

dim3 grid(4, 1, 1)表示的是一个grid由四个一维block组成，不要混淆为有4个grid

block(8, 1, 1)表示的是一个block由1维thread组成，1乘8的一维分布，不要混淆为有8个block

block(4, 2, 1)表示的是一个block由2维thread组成，4乘2的二维分布，不要混淆为有4个长度为2的block

这就是为什么没有threadDim.x，threadDim.y，threadDim.z的原因
```

```
本质上是在网格索引，所以从x方向开始查找和从y方向开始查找结果是一样的。对于一个block(4, 2, 1)而言
int tid = threadIdx.x*blockDim.y + threadIdx.y;
int tid = threadIdx.y*blockDim.x + threadIdx.x;
以上两个索引结果等价

在索引的时候要计算，不要自己套 公式，例如：
dim3 grid(8, 1, 1), block(2, 1, 1); 的情况，索引就是：
int tid = blockIdx.x * blockDim.x + threadIdx.x;而不是原文中的blockIdx.x * gridDim.x + threadIdx.x， 原文只是恰巧gridDim.x = blockDim.x  = 4 而已

在dim3 grid(2, 2, 1), block(2, 2, 1);   这种block和thread都是二维分布的情况下：
    int tid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.x * blockDim.y)
              + threadIdx.y*blockDim.x + threadIdx.x;
    int tid = (blockIdx.x * gridDim.y + blockIdx.y)*(blockDim.x * blockDim.y)
              + threadIdx.x*blockDim.y + threadIdx.y;
```

```
1、 grid划分成1维，block划分为1维

    int threadId = blockIdx.x *blockDim.x + threadIdx.x;


2、 grid划分成1维，block划分为2维

    int threadId = blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;


3、 grid划分成1维，block划分为3维

    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                       + threadIdx.z * blockDim.y * blockDim.x
                       + threadIdx.y * blockDim.x + threadIdx.x;


4、 grid划分成2维，block划分为1维

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;


5、 grid划分成2维，block划分为2维

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
                       + (threadIdx.y * blockDim.x) + threadIdx.x;


6、 grid划分成2维，block划分为3维

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                       + (threadIdx.z * (blockDim.x * blockDim.y))
                       + (threadIdx.y * blockDim.x) + threadIdx.x;


7、 grid划分成3维，block划分为1维

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                     + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;


8、 grid划分成3维，block划分为2维

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                     + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
                       + (threadIdx.y * blockDim.x) + threadIdx.x;


9、 grid划分成3维，block划分为3维

    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                     + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                       + (threadIdx.z * (blockDim.x * blockDim.y))
                       + (threadIdx.y * blockDim.x) + threadIdx.x;
```

**Caffe 框架中都是将数据拉伸为一维，图像坐标计算如下**

```
#define CUDA_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)


```
