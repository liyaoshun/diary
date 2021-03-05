# Pytorch 相关问题

## torch.backends.cudnn.* 配置问题
[Link](https://zhuanlan.zhihu.com/p/141063432?from_voters_page=true)
```
1. 在训练网络的时候如果想每次训练得到的结果是一样的话需要配置相同的参数，使用相同的网络结构、学习率、迭代次数、batch size，然后还需要固定随机种子，cuda的话需要设置如下：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 404
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
2. 按照上方的配置存在一个问题，使用了deterministic = True会导致模型运行速度特别慢
```

## pytorch高版本训练的模型使用低版本加载问题
```
情形：假设我们现在使用pytorch1.7训练好了pth模型，然后需要在嵌入式平台进行部署，但是嵌入式平台只能用tensorrt5.1.5，此时pytorch1.7对于此环境下来讲就显得高了，需要将pytorch版本降低到pytoch1.1.0，然后再转存模型到 pth->onnx->trt。
1. 如果直接使用pytorch1.1.0加载pytorch1.7训练好的模型会报错。解决方案如下代码：
    切换pytorch版本为1.7执行下面代码
    model_state_file = “xxx/xxx.pth”
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    torch.save(pretrained_dict, 'xxx/xxxx_new.pth', _use_new_zipfile_serialization=False)
    切换pytorch版本为1.1执行下面代码
    model_state_file = 'xxx/xxxx_new.pth'
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    model = Gmodel()
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    net = model
    torch.onnx.export(net,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                output_path,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=9,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['inputx'],   # the model's input names
                output_names = ['outputy'], # the model's output names
                verbose=True,
                )
    Tips：opset_version=9 一定使用9,如果是11的话会到时upsample出错。
```
