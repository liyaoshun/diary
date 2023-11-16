# **polygraphy优化输出**
https://zhuanlan.zhihu.com/p/436017991
https://www.stubbornhuang.com/1737/
https://blog.csdn.net/TracelessLe/article/details/120656484

## **定位误差较大位置**
### 排除onnx转换错误

定性: onnxruntime来运行转换后的onnx权重，输出结果与pytorch输出结果对比.

定量: 保持相同的数据预处理方式，对比onnx和pytorch模型的输出，可以使用np.testing.assert_allclose 来进行精度校验
通过上述方法可以发现onnx模型转换是正确的，那说明tensorrt的转换出了问题

### Polygraphy逐层分析定位错误节点

polygraphy是tensorrt提供的分析工具，可以对比onnx与tensorrt模型各个层输出的差异来定位问题
```
分析步骤如下:
1. 分别保存onnx和tensorrt模型的全部的网络层输出
2. 通过循环遍历各个层，找到精度无法匹配的层节点位置
3. 分析该位置对应的onnx节点找到可能的错误，如tensorrt不支持的节点
```

在终端输入如下的指令保存onnx各层的输出
polygraphy run yolov5s.onnx --onnxrt --onnx-outputs mark all --save-results=onnx_out.pkl

终端输入如下的指令保存tensorrt各个层的输出
polygraphy run yolov5/yolov5s.onnx --trt --validate --trt-outputs mark all --save-results=trt_out.pkl

注意：这里的mark all表示保存各个层的输出，可以通过指定层名称来只保存对应的输出

逐层对比精度
```
import pickle
import numpy as np

f = open('onnx_out.pkl','rb')
info_onnx = pickle.load(f)
# print(info_onnx)
f = open('trt_out.pkl','rb')
info_trt = pickle.load(f)
# print(info)
runners_trt = list(info_trt.keys())
runners_onnx = list(info_onnx.keys())

print('onnx:', len(info_onnx.__getitem__(runners_onnx[0])[0]))
print('tensorrt:', len(info_trt.__getitem__(runners_trt[0])[0]))

for layer in info_onnx.__getitem__(runners_onnx[0])[0]:
    if layer in info_trt.__getitem__(runners_trt[0])[0]:    # 只分析相同名称的节点输出
        print('--------------------------')
        print(layer, info_onnx.__getitem__(runners_onnx[0])[0][layer].shape, info_trt.__getitem__(runners_trt[0])[0][layer].shape)
        onnx_out = info_onnx.__getitem__(runners_onnx[0])[0][layer]
        trt_out = info_trt.__getitem__(runners_trt[0])[0][layer]
        np.testing.assert_allclose(onnx_out, trt_out, 0.0001, 0.0001)
```