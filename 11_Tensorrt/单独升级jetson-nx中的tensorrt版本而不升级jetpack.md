# **单独升级jetson-nx中的tensorrt版本而不升级jetpack**

## **单独升级版本**
参考: https://blog.csdn.net/weixin_44511447/article/details/130579370

1. 获取目标版本号L4T 32.7.1
2. 在文本编辑器中打开 apt 源配置文件 
sudo vim /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

更改配置，将文件中的r32.5更改为r32.7（与目标版本号对应）
deb https://repo.download.nvidia.com/jetson/common r32.7 main
deb https://repo.download.nvidia.com/jetson/t210 r32.7 main
sudo apt update

3. 在pc上使用sdkmanager下载下面所需要的包，选择型号和版本后，下载到本地文件夹即可

4. 将tensorrt的依赖包复制到板子上面，安装(示例)

dpkg -i libnvinfer7_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvinfer-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvinfer-plugin7_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvinfer-plugin-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvonnxparsers7_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvonnxparsers-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvparsers7_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvparsers-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvinfer-bin_7.2.0-1+cuda10.2_arm64.deb
dpkg -i libnvinfer-doc_7.2.0-1+cuda10.2_all.deb
dpkg -i libnvinfer-samples_7.2.0-1+cuda10.2_all.deb
dpkg -i tensorrt_7.2.0.14-1+cuda10.2_arm64.deb
dpkg -i python-libnvinfer_7.2.0-1+cuda10.2_arm64.deb
dpkg -i python-libnvinfer-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i python3-libnvinfer_7.2.0-1+cuda10.2_arm64.deb
dpkg -i python3-libnvinfer-dev_7.2.0-1+cuda10.2_arm64.deb
dpkg -i graphsurgeon-tf_7.2.0-1+cuda10.2_arm64.deb
dpkg -i uff-converter-tf_7.2.0-1+cuda10.2_arm64.deb


## 