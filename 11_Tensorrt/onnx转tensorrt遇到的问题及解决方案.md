# **onnx转tensorrt遇到的问题及解决方案**

## **pad 值为0报错**
```
报错信息:ERROR: /media/robot/4T/10_tensorrt/TensorRT7.2/parsers/onnx/builtin_op_importers.cpp:2260 In function importPad:
[8] Assertion failed: mode == "constant" && value == 0.f && "This version of TensorRT only supports constant 0 padding!"
解决方案:  升级trt版本. -> 8.4

issue: https://github.com/NVIDIA/TensorRT/issues/1019
```


## **单个算子别分解为多个算子**
例如: gelu被分解为tanh等不同算子的组合。
TRT部署优化细节: https://ownlu.com/524-2/

```
LayerNorm: 在Onnx中为多个算子的组合，用Onnx-Graphsurgeon将这些节点合并为一个LayerNorm节点。同时，书写Plugin，通过Cuda核函数实现LayerNorm的计算逻辑。在Onnx转TRT的过程中，TRT会自动寻找与LayerNorm的同名Plugin进行解析。

```
## **where算子在opset_version 11支持不好**
说明: pytorch中where有两个用法
1. torch.where(a>0, 1, 0),用于筛选tensor中满足条件的设置为1，不满足的设置为0.
2. x,y = torch.where(a), 返回a中大于0的所有坐标信息。

用法1在tensorrt中满足条件，但是用法2在进行转换onnx的时候会将操作使用nonzero代替，此时tensorrt需要8.5以后才能支持的很好.

修改demo:
```
attn_mask: 24* 100 * 192

condition_ = attn_mask.sum(-1) == attn_mask.shape[-1]       : 24 * 100

原始操作:  attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

修改后的操作: 
condition_ = attn_mask.sum(-1) == attn_mask.shape[-1] 
condition_inv = ~condition_
mask_all_ = condition_inv.repeat(192, 1, 1).permute(1, 2, 0)
attn_mask = mask_all_ * attn_mask

验证: 使用 a.equal(b)  或者 a.eq(b)

```


## **InstanceNormalization onnx转trt不支持动态**
```
报错信息: [8] Assertion failed: !isDynamic(tensorPtr->getDimensions()) && "InstanceNormalization does not support dynamic inputs!"
issue: https://github.com/onnx/onnx-tensorrt/issues/374
果当前版本是7.1.3还需要在builtin_op_importers.cpp中将trt7.2中的DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)拷贝到7.1.3中再重新编译和替换原来的动态库
```

## **F.interpolate不支持使用scale_factor**
当前错误在trt7.1.3中出现，其它版本没有测试
报错信息:[8] Assertion failed: scales.is_weights() && "Resize scales must be an initializer!"
解决方案:将resize后的大小固定下来

直接修改onnx模型方案: https://zhuanlan.zhihu.com/p/456570769
```
1. 安装工具
snap install netron
pip install onnx-simplifier
pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
之后，Netron 查看模型 Resize不支持节点名

修改步骤如下:
1.onnx模型simple简化

```

## **Add_854 版本问题**
报错信息: [8] Assertion failed: convertOnnxWeights(initializer, &weights, ctx)
Found unsupported datatype (11) when importing initializer: onnx::Add_854
解决方案: onnx版本太新，需要降低版本
