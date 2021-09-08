# **关于多 batch 推理的相关问题**

问题描述：在使用 tensorrt 对语义分割模型进行不同 batch 推理耗时成线性增长

## **模型转换方式**

### **使用开源 onnx2trt**

特点： 编写 layer 插件比较方便。

### **使用 Tensorrt 自带的转换工具**

特点： 一键编译，同时还可以用于测试模型耗时指标。同时也可以方便的编写不同的 layer 插件。

## **engine 的导出方式**

在pc上测试的耗时展示

| 设备 | 模型         | model | batch | gpu 计算耗时/ms |
| :--- | :----------- | :---- | :---- | :-------------- |
| pc   | ddrnet23_ocr | fp16  | 1     | 1.70512         |
| pc   | ddrnet23_ocr | fp16  | 2     | 2.70069         |
| pc   | ddrnet23_ocr | fp16  | 4     | 4.78706         |
| pc   | ddrnet23_ocr | fp16  | 8     | 9.03271         |
| pc   | ddrnet23_ocr | fp16  | 16    | 16.1414         |


### **固定 batch 方式**

1. 在将 onnx 模型转换为 engine 的时候就固定 batch 的大小，这样在推理的时候需要使用固定的 batch 进行推理。缺点是模型不够灵活，在不同条件和要求下需要重新转换 engine。

eg:

```
    dump_input = torch.rand(
        (2, 3, 480, 640)
    )

    torch.onnx.export(net,               # model being run
                    dump_input,                         # model input (or a tuple for multiple inputs)
                    output_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['inputx'],   # the model's input names
                    output_names = ['outputy_class','outputy_score'], # the model's output names
                    #   output_names=['outputy'],  # the model's output names
                    verbose=True,
                    )
```

### **动态 batch 方式**

1. 基于固定 batch 的缺点，尝试使用动态 batch 的方式进行 engine 的转换，使得在可以适用不同 batch 的要求。

eg:

```
    x = torch.onnx.export(net,  # 待转换的网络模型和参数
                          torch.randn(2, 3, 480, 640),  # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                          export_dynamic_onnx_file,  # 输出文件的名称
                          verbose=False,  # 是否以字符串的形式显示计算图
                          input_names=["inputx"],
                          # + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                          output_names=["outputy_class", "outputy_score"],  # 输出节点的名称
                          opset_version=11,  # onnx 支持采用的operator set, 应该和pytorch版本相关
                          do_constant_folding=True,  # 是否压缩常量
                          dynamic_axes={"inputx": {0: "batch_size"}, "outputy_class": {0: "batch_size"}, "outputy_score" :{0: "batch_size"}}
                          # 设置动态维度，此处指明input节点的第0维度可变，命名为batch_size
                          )
```

## **遇到的问题**

不论是使用动态还是静态 batch 的时候在 bs 从 1 增大到 2 的时候，推理时间都会增加 1.2~2 倍之间。

## **github 中讨论的 issue**

[issue](./issue/trtexec_dynamic_batch_size_Issue976.pdf)
