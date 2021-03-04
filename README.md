# diary
## 2021.03.04
```
情形：部署pytorch训练后的模型，将.pth模型使用以下的代码转为.onnx的模型，接着使用onnx2trt将onnx模型转为trt模型，最后在tx2上部署。

torch.onnx.export(net,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                output_path,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=11,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['inputx'],   # the model's input names
                output_names = ['outputy'], # the model's output names
                verbose=True,
                )

Tips.
1. tx2上分为tensorrt5.1.5和tensorrt7.2.2两个版本，使用tensorrt5.1.5的时候需要使用pytorch1.1.0转存onnx模型,tensorrt7.2.2可以使用pytorch1.7转存模型。
3.tensorrt5.1.5没有prelu激活层，所以需要自己修改。
4.tensorrt7.2.2版本的完整项目在github:https://github.com/liyaoshun/trt_deploy.git (私人仓库) ^.^ ^.^ ^.^

```

```
pytorch模型转存onnx注意事项：
1. 变量需要使用常量形式，如包含有.size或者.shape的语句需要使用torch.tensor(.size(i)).item()
2. F.interpolate 在使用双线性的时候需要设置align_corners=False
3. 使用多卡训练的模型然后再使用单卡测试的时候，BN在转存的时候使用torch.nn.BatchNorm2d代替torch.nn.SyncBatchNorm。
```