# **转换问题及解决方案**
 涉及到动态问题: 因为tensorrt是预分配固定空间框架，如果模型中存在动态算子的话就基本上不支持。当然动态输入和输出是支持的.

## **AdaptiveAvgPool2d在onnx和tensorrt中不支持**
```
原因: 不支持的主要原因是自适应均值池化需要动态的计算池化核大小和pad大小.(主要是动态问题)

解决方案1: 在训练阶段可以使用AdaptiveAvgPool2d操作，但是在部署的时候将AdaptiveAvgPool2d手动替换为AvgPool2d，同时需要根据输入shape计算kernel_size和stride。

解决方案2:使用下方代码替换，可以参考https://github.com/hustvl/SparseInst/blob/main/onnx/convert_onnx.py中做法进行替换模块.
class PyramidPoolingModuleONNX(nn.Module):

    def __init__(self, in_channels, channels, input_size, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, input_size, pool_size)
             for pool_size in pool_sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(pool_sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, input_size, pool_size):
        stride_y = math.floor((input_size[0] / pool_size))
        stride_x = math.floor((input_size[1] / pool_size))
        kernel_y = input_size[0] - (pool_size - 1) * stride_y
        kernel_x = input_size[1] - (pool_size - 1) * stride_x
        prior = nn.AvgPool2d(kernel_size=(
            kernel_y, kernel_x), stride=(stride_y, stride_x))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(
            input=F.relu_(stage(feats)), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out
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
https://www.cnblogs.com/zhongzhaoxie/p/16692999.html


```

## **NonZero算子不支持**
```
报错信息: KeyError: 'nonzero_numpy'
[8] Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?"

issue1: https://github.com/pytorch/vision/pull/2314
issue2: https://github.com/NVIDIA/TensorRT/issues/2285

原因: 此算子的主要功能是提取标量中非零值的索引，它的返回值的长度是可变的,涉及到动态问题。
onnx转tensorrt支持方案: https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md  但是需要升级tensorrt版本到8.6.

torch.nonzero()和torch.index_select()，筛选张量中符合某种条件的元素。(NonZero是TensorRT中明确说明不支持的算子，但是index_select并没指出，可以尝试替换)

三方解决方案:https://blog.csdn.net/xuanwu_yan/article/details/111463822

```

## **torch.nn.functional.pad算子不支持**
```
原因: 转换onnx模型的时候不会报错误，但是会报警告。
报错信息: [8] Assertion failed: mode == "constant" && value == 0.f && "This version of TensorRT only supports constant 0 padding!"
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
issue中的解决方案:https://github.com/huggingface/transformers/pull/25297/files/bbcfbd1f3410372911fe6b59b93ac25cd7f3cf45#diff-f3fcabe94246623f20f3e272bdd94d3ee9f5d749736072e865109d34b8c74247
1.
            binaries_masks = torch.einsum("lbqc,   bchw -> lbqhw", mask_embeddings, pixel_embeddings)
            # (num_embeddings, batch_size, num_queries, num_channels, 1, 1)
            mask_embeddings = mask_embeddings.unsqueeze(-1).unsqueeze(-1)
            # (1, batch_size, 1, num_channels, height, width)
            pixel_embeddings = pixel_embeddings.unsqueeze(0).unsqueeze(2)
            # (num_embeddings, batch_size, num_queries, height, width)
            binaries_masks = (mask_embeddings * pixel_embeddings).sum(dim=3)

2.
            masks_queries_logits = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)
            # (batch_size, num_queries, num_channels, 1, 1)
            mask_embeddings = mask_embeddings.unsqueeze(-1).unsqueeze(-1)
            # (batch_size, 1, num_channels, height, width)
            pixel_embeddings = pixel_embeddings.unsqueeze(1)
            # (batch_size, num_queries, num_channels, height, width)
            masks_queries_logits = (mask_embeddings * pixel_embeddings).sum(dim=2)

3.



```

## **torch.tensor.T转onnx numpy_T报错**
```
错误提示:torch.onnx.symbolic_registry.UnsupportedOperatorError: Exporting the operator ::numpy_T to ONNX opset version 12 is not supported. Please feel free to request support or submit a pull request on PyTorch GitHub.

issue1: https://github.com/pytorch/pytorch/pull/79269
issue2: https://github.com/pytorch/pytorch/issues/51183
使用.transpose(0, 1)代替.T
```

## **torch.where 算子不支持**
```
报错信息: TypeError: where() missing 2 required positional arguments: 'self' and 'other' (occurred when translating where)
```

## **torch.masked_selected算子不支持**
```
报错信息: KeyError: 'masked_select'，
```

## **cumsum算子不支持**

```
tensorrt 7.1 不支持，需要修改，用rangge替代，但是在转onnx后会将其使用where等操作替代.
https://github.com/NVIDIA/TensorRT/blob/release/9.0/demo/HuggingFace/BLOOM/export.py#L55-L58
```



