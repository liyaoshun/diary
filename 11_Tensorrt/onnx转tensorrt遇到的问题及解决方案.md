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

验证: 使用 a.equal(b)。

```