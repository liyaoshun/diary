# Pytorch 相关问题

## ***torch.backends.cudnn.x 配置问题***
[LINK](https://zhuanlan.zhihu.com/p/141063432?from_voters_page=true)
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

## **pytorch高版本训练的模型使用低版本加载问题**
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

## **pytorch 多GPU使用问题**
[LINK](https://zhuanlan.zhihu.com/p/86441879)
```
多GPU分布式训练分为单机多GPU和多机多GPU两种类型，官网的解释如下图所示:
```
![Image text](images/mul_gpu.png)

### **单机多卡并行训练**
1. torch.nn.DataParallel
    ```
    使用下面的代码将模型分发到不同GPU上.
    model = nn.DataParallel(model)
    model = model.cuda()
    ```
2. 如何平衡DataParallel带来的显存使用不平衡的问题
   官方给的解决方案就是使用 DistributedDataParallel来代替 DataParallel。但是这个函数也存在显存分配不均衡的问题，下方链接是一个平衡策略：
   [Github](https://github.com/Link-Li/Balanced-DataParallel)
   
3. torch.nn.parallel.DistributedDataParallel
   ```
    单机多卡初始化：
    torch.distributed.init_process_group(backend="nccl", init_method="env://",)
    model = torch.nn.parallel.DistributedDataParallel(
                                                    model,
                                                    find_unused_parameters=True,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank
                                                    )
    此时启动训练代码命令如下：
    python -m torch.distributed.launch --nproc_per_node=2 train.py 
   ```

**Tips**:
    ```
    1. os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    2. 注: 这里如果使用了argparse, 一定要在参数里面加上--local_rank, 否则运行还是会出错的
    ```

### **多机多gpu训练**
    ```
    当前没有使用多机训练代码
    ```

## **paddlepaddle权重转pytorch格式**
主要参考下方代码的实现
[LINK](https://github.com/maomaoyuchengzi/paddlepaddle_param_to_pyotrch)

最重要的地方需要注意的是在转conv的时候需要注意是否是有biass存在，然后在转bn的时候需要转weight、bias、running_mean、running_var这四个权重。如果两个框架编写模型时候的命名不相同的话需要主要其转换的规则。主要是有时候不同框架的权重的保存顺序不同，这个时候就需要进行特殊的处理。还有就是有时候有不同的中间的key名称，需要使用跳过的方式处理。

主要的代码如下：
```
def _load_state():
    dst = "xxx_pretrained.pdparams"
    state = fluid.io.load_program_state(dst)
    return state

# 使用pytorch构建网络结构，用来接受paddle转过来的权重信息。
def get_pytorch_model():
    return Model()

backbone = get_pytorch_model()
state_pp = list(_load_state().keys())

# 下边的替换是基于名称相同时候的操作
for n, m in backbone.named_modules():
    if isinstance(m, BatchNorm2d):
        //***
    elif isinstance(m, Conv2d):
        m.weight.data.copy_(torch.FloatTensor(state_pp[n]))
    else:
        print(n)

```