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

![Image text](images/derivative.svg)
