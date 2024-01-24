# 多输入输出的动态tensorrt导出问题

```
此记录以转superglue为实验
```

## SuperGlue转tensorrt
### 在转模型时会遇见以下问题
```
1. def attention(query, key, value): 函数中的torch.einsum不能导出。修改后的函数见页面最后附录

2. 前向推理中的torch.einsum也会出现问题，需要根据'bdn,bdm->bnm'规则进行修改。 https://github.com/huggingface/transformers/pull/25297/files/bbcfbd1f3410372911fe6b59b93ac25cd7f3cf45#diff-f3fcabe94246623f20f3e272bdd94d3ee9f5d749736072e865109d34b8c74247

3. log_optimal_transport 函数中的cat由于b, m, n = scores.shape会产生不能对齐维度问题。同时也不能修改为b, m, n = torch.tensor(scores.shape).tolist()，这样会导致到出出的模型中有固定维度，也就是b, m, n值会被转为常量。

4. arange_like函数中的cumsum函数在tensorrt7.2.1.6中也不支持，当不是动态转换模型的时候可以使用onnxsim优化，使其可以导出为tensorrt模型。
```

## SuperPoint转tensorrt
### 在转模型时会遇见以下问题
```
1. torch.nn.functional.grid_sample不能导出，tensorrt不支持。使用旷视的bilinear_grid_sample替换。

2. torch.gather在tensorrt7.2.1.6中不支持，需要编写插件，插件文件在./plugins/gataherPlugin

3.torch.nn.functional.normalize 需要使用X = X.div(X.norm(p=2, dim=1, keepdim=True))分布操作来替代

4. NonZero操作在tensorrt7.2.1.6中不被支持

5. 报错信息： Assertion failed: inputs.at(2).is_weights() && "Clip max value must be an initializer!"   。 由desc.norm中后面链接的clamp_min(1e-12).expand_as(desc)导致

```

## ** trtexec转TensorRT命令（动态输入，多输入）**
```
onnx转tensorrt:
1. 如果是固定输入:
trtexec --onnx=vovnet.onnx --saveEngine=vovnet.trt --fp16 --workspace=10240
trtexec --onnx=CrossFormer-T.onnx --saveEngine=CrossFormer-T.trt --fp16 --workspace=10240

2. 如果是动态输入(单输入 & 多输入):
trtexec --onnx=nextvit_pp.onnx --saveEngine=nextvit_pp.trt --fp16 --workspace=10240 --explicitBatch --minShapes=voxels:5000x20x5,num_points:5000,coors:5000x4 --optShapes=voxels:20000x20x5,num_points:20000,coors:20000x4 --maxShapes=voxels:30000x20x5,num_points:30000,coors:30000x4


/usr/src/tensorrt/bin/trtexec --onnx=./superpoint_lightglue.onnx --saveEngine=./superpoint_lightglue_fp16.trt  --minShapes=kpts0:1x300x2,kpts1:1x300x2,desc0:1x300x48,desc1:1x300x48,matches0:300x2,mscores0:300  --optShapes=kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x48,desc1:1x512x48,matches0:512x2,mscores0:512 --maxShapes=kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x48,desc1:1x512x48,matches0:512x2,mscores0:512 --explicitBatch --workspace=4096 --fp16 --verbose


如果转Tensor RT时报错: invalid memory access, Core dumped.
将--fp16去掉，尝试是否能转成功。

```










## 附录
```
attention 用于导出tensorrt

def attention(query, key, value):
    dim = query.shape[1]
    _query  = query.permute(0, 2, 3, 1).contiguous()
    _key    = key.permute(0, 2, 1, 3).contiguous()
    scores = torch.matmul(_query, _key) / dim**.5
    # scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    _value = value.permute(0, 2, 3, 1).contiguous()
    _out = torch.matmul(prob, _value)
    out = _out.permute(0, 3, 1, 2).contiguous()
    return out, prob
```