当前仓库下的 gather 插件有如下缺点:

```
1. 没有编写half精度的cuda代码
2. 使用此插件的时候输入数据类型做了严格的限制，输入数据类型必须为float32,index 数据必须为int64类型.不然会出现如下错误提示（Could not find any supported formats consistent with input/output data types）
```

```
superpoint onnx 转 tenorrt:
Assertion failed: inputs.at(2).is_weights() && "Clip max value must be an initializer!"

发现是F.normalize(descriptors, p=2, dim=1)导致,使用如下代码替换：
desc_norm = descriptors.norm(2, 1, keepdim=True)
descriptors =  descriptors /desc_norm

```

```
https://zhuanlan.zhihu.com/p/572548014?utm_id=0

While parsing node number 211 [Pad]:
ERROR: /media/robot/4T/10_tensorrt/TensorRT7.2/parsers/onnx/builtin_op_importers.cpp:2248 In function importPad:
[8] Assertion failed: inputs.at(1).is_weights()


其他解决方案： 使用onnxsim进行网络结构的精简
```
