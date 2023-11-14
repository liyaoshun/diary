# **转换问题及解决方案**
 涉及到动态问题: 因为tensorrt是预分配固定空间框架，如果模型中存在动态算子的话就基本上不支持。当然动态输入和输出是支持的.

## **AdaptiveAvgPool2d在onnx和tensorrt中不支持**
```
原因: 不支持的主要原因是自适应均值池化需要动态的计算池化核大小和pad大小.(主要是动态问题)
解决方案: 在训练阶段可以使用AdaptiveAvgPool2d操作，但是在部署的时候将AdaptiveAvgPool2d手动替换为AvgPool2d，同时需要根据输入shape计算kernel_size和stride。
```

## **Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.**
```
原因: 可能是torch.nn.functional.pad算子不支持
解决方案: 和下面的torch.nn.functional.pad算子解决方案一致.
```

## **Gelu 激活函数onnx个tensorrt不支持**
```
解决方案: 需要编写tensorrt插件和自定义pytorch gelu模块。
参考资料: https://blog.csdn.net/weixin_45878768/article/details/128149343
```

## **NonZero算子不支持**
```
原因: 此算子的主要功能是提取标量中非零值的索引，它的返回值的长度是可变的,涉及到动态问题。

torch.nonzero()和torch.index_select()，筛选张量中符合某种条件的元素。(NonZero是TensorRT中明确说明不支持的算子，但是index_select并没指出，可以尝试替换)

暂时还没有很好的解决方案.
```

## **torch.nn.functional.pad算子不支持**
```
原因: 转换onnx模型的时候不会报错误，但是会报警告。
解决方案: 改变padding操作的实现方式：zero padding的话先torch.zeros得到需要进行pad的“补丁”tensor，然后再与需要被padding的tensor进行concat操作即可。
```

## **时torch.nn.functional.fold算子不支持**
```
报错信息: RuntimeError: Exporting the operator col2im to ONNX opset version 11 is not supported
issue: https://github.com/KinWaiCheuk/nnAudio/issues/102

```

## **F.conv2d问题**
```
F.conv2d，在PyTorch到ONNX步骤能正常导出，但是从ONNX到TensorRT步骤则会报错。
(实际在pytorch转onnx这步就已经开始报错：onnx export of convolution for kernel of unknown shape)

```

## **torch.einsum 算子替换**
因为tenosrrt中暂时没有实现einsum算子，需要使用torch自带的算子替换。
```

```

## **torch.tensor.T转onnx numpy_T报错**
```
错误提示:torch.onnx.symbolic_registry.UnsupportedOperatorError: Exporting the operator ::numpy_T to ONNX opset version 12 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.

issue1: https://github.com/pytorch/pytorch/pull/79269
issue2: https://github.com/pytorch/pytorch/issues/51183
使用.transpose(0, 1)代替.T
```