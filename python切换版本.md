切换ubuntu18.04 默认python
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

sudo update-alternatives --config python

同时在zshrc中添加 alias python='/usr/bin/python'  在 ubuntu20.04里面添加alias会导致在conda环境里面调用的python一直是系统自带的python版本

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run
