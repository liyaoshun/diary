# jetson nx 464 安装conda

##  安装流程
```
1. 下载Miniforge3-4.12.0-0-Linux-aarch64.sh并安装  报错解决:sudo rm -r ~/.condarc
2. 创建虚拟环境:conda create -n yolov9 python==3.6 | conda activate yolov9
3. 安装pytorch1.10.0: 下载whl包https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048然后pip install ./torch-1.10.0-cp36-cp36m-linux_aarch64.whl
4. 安装torchvision: git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision, cd torchvision && python setup.py install

```
