切换ubuntu18.04 默认python
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

sudo update-alternatives --config python

同时在zshrc中添加 alias python='/usr/bin/python'