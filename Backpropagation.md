# Overview
https://en.wikipedia.org/wiki/Backpropagation
```
Backpropagation computes the gradient in weight space of a feedforward neural network, with respect to a loss function. Denote:
1. x: input (vector of features)
2. y: target output
    For classification, output will be a vector of class probabilities (e.g., (0.1,0.7,0.2), and target output is a specific class, encoded by the one-hot/dummy variable (e.g.,(0,1,0)).
3. C: loss function or "cost function"
    For classification,this is usually cross entropy,while for regression it is usually squared error loss
4. L: the number of layers
5. w_{jk}^{l}: the weights between layer l-1 and l, where w_{jk}^{l} is the weight between the k-th node in layer l-1 and the j-th node in layer l
6.  f^{l}: activation functions at layer l
    For classification the last layer is usually the logistic function for binary classification, and softmax (softargmax) for multi-class classification, while for the hidden layers this was traditionally a sigmoid function (logistic function or others) on each node (coordinate), but today is more varied, with rectifier (ramp, ReLU) being common.
```

## Matrix multiplication
```
Given an input–output pair (x,y), the loss is:
给一个输入输出对样本，计算损失值公式如下：
```
![Image text](images/forward_loss.svg)
```
The derivative of the loss in terms of the inputs is given by the chain rule; note that each term is a total derivative, evaluated at the value of the network (at each node) on the input x:
在损失函数上对输入变量x求导公式如下：
```
![Image text](images/derivative.svg)
![Image text](images/derivative_1.svg)

```
The gradient  is the transpose of the derivative of the output in terms of the input, so the matrices are transposed and the order of multiplication is reversed, but the entries are the same:
```
![Image text](images/gradient.svg)


---
Introducing the auxiliary quantity ![Image text](images/auxiliary_quantity.svg) for the partial products (multiplying from right to left), interpreted as the "error at level l" and defined as the gradient of the input values at level l:
![Image text](images/auxiliary_quantity_1.svg)

```
The gradient of the weights in layer l is then:
```
![Image text](images/gradient_w.svg)

The ![Image text](images/auxiliary_quantity.svg) can easily be computed recursively as:

![Image text](images/delta_l1.svg)

The gradients of the weights can thus be computed using a few matrix multiplications for each level; this is backpropagation.Compared with naively computing forwards (using the ![Image text](images/auxiliary_quantity.svg) for illustration):

![Image text](images/delta_all.svg)

there are two key differences with backpropagation: