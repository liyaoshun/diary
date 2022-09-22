# pytorch 转 tensorrt 精度损失研究

```
对比方法介绍：https://blog.csdn.net/TracelessLe/article/details/120656484
背景：使用superpoint 和 superglue进行slam建图。
```

## superpoint 在转 tensorrt 时候遇见的问题

```
nonzeros 不被支持
```

```
对比每一层的输出差别：
POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB=0表示结果交换到硬盘上，不放在内存中，mark all 表示对比所有节点结果

POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB=0 polygraphy run superpoint_128dim_300_repvgg_0921_repvgg_acc_float32_simple.onnx  --trt --onnxrt --tactic-sources CUBLAS  --rtol 1e-05 --atol 1e-03  --trt-outputs mark all --onnx-outputs mark all

```

```


# 使用默认参数
polygraphy run polygraphy_debug.engine --trt

# 加载onnx推理得到的结果，并设置绝对误差
polygraphy run polygraphy_debug.engine --trt --load-outputs onnx_res.json --abs 1e-4

# 使用fp16和cublas策略库生成engine，并与onnx推理得到的结果作比较
polygraphy run net_bs8.onnx --trt --onnxrt --tactic-sources CUBLAS --fp16 --rtol 1e-03 --atol 1e-03

# 保存 data_loader生成的输入和运行得到的结果
polygraphy run net_bs8.onnx --onnxrt  --val-range [0,1] --save-inputs  net_input.json --save-outputs onnx_res.json
# 对onnx推理得到的结果进行保存，同时运行trt engine，并与onnx推理得到的结果作比较
polygraphy run net_bs8.onnx --onnxrt --load-inputs  gpen_input.json --save-outputs onnx_res.json
polygraphy run net_bs8_fp32.engine --model-type engine --trt --load-outputs onnx_res.json --abs 1e-4
--load-inputs  net_input.json

# 采用二分策略，使用fp16和cublas策略库按层数迭代生成engine，并加载onnx推理得到的结果作比较，以判断设置fp16时误差出现的层的范围
polygraphy debug precision net_bs8.onnx --fp16 --tactic-sources cublas --check polygraphy run polygraphy_debug.engine --trt --load-outputs onnx_res.json --abs 1e-1





# 使用polygraphy debug precision工具搜索Layer精度设置，判断FP16的engine运行结果与onnx结果的误差是否在误差范围内，不移除运行中生成的engine文件
polygraphy debug precision net_bs8.onnx --fp16 --tactic-sources cublas --check polygraphy run polygraphy_debug.engine --trt --load-inputs  net_input.json --load-outputs onnx_res.json --abs 1e-1 --no-remove-intermediate

# POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB=0表示结果交换到硬盘上，不放在内存中，mark all 表示对比所有节点结果
POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB=0 polygraphy run net_bs8.onnx --trt --onnxrt --tactic-sources CUBLAS --fp16 --rtol 1e-03 --atol 1e-03  --trt-outputs mark all --onnx-outputs mark all


# 比较单个或者多个Tensor的结果
polygraphy run net_bs8.onnx --onnxrt --trt --tactic-sources CUBLAS --fp16 --rtol 1e-03 --atol 1e-03 --trt-outputs 151 --onnx-outputs 151

polygraphy debug run net_bs8.onnx --onnxrt  --check polygraphy run net_bs8.engine --trt --tactic-sources CUBLAS --fp16 --rtol 1e-03 --atol 1e-03 --trt-outputs 153 197 --onnx-outputs 153 197


# 检查模型
polygraphy inspect model net_bs8.engine --mode=basic --display-as=trt
polygraphy inspect model net_bs8.onnx --mode=full --display-as=trt

```

```
当tensort解析其中包含有自己编写的插件的时候会有报错信息：

dvs/p4/build/sw/rel/gpgpu/MachineLearning/myelin_daisy_TRT7.2/src/compiler/./ir/operand.h:157: myelin::ir::tensor_t*& myelin::ir::operand_t::tensor(): Assertion `is_tensor()' failed.
```

## repvgg 网络会丢失精度，不太适合使用在精度较高的任务中
