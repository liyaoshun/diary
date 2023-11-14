# **转换问题及解决方案**

## **AdaptiveAvgPool2d在onnx和tensorrt中不支持**
```
原因: 不支持的主要原因是自适应均值池化需要动态的计算池化核大小和pad大小.(主要是动态问题)
解决方案: 在训练阶段可以使用AdaptiveAvgPool2d操作，但是在部署的时候将AdaptiveAvgPool2d手动替换为AvgPool2d，同时需要根据输入shape计算kernel_size和stride。
```

## **Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.**
```
原因:
解决方案:
```