## 拷贝宿主机文件到 container

nvidia-docker cp /**/**/demo.txt mycontainer:/sss/sss/
提交修改，保证下次打开修改了的文件还存在。
sudo docker commit -m "描述内容" -a "author name" 容器 id images_name:tag

## **2021.03.19**

**Docker 环境搭建：** [Link1](https://docs.docker.com/engine/install/ubuntu/)
[Link2](https://www.jianshu.com/p/49e8f814d6e0)

1. Step 1. ubuntu18.04 上安装 docker

   ```
    sudo apt-get update

    sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
    //添加Docker官方GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    //设置Docker稳定版仓库
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    //添加仓库后，更新apt源索引
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    apt-cache madison docker-ce
   ```

   ```
   开启docker服务:sudo service docker start
   关闭docker服务:sudo service docker stop
   重启docker服务:sudo service docker restart

   docker状态查看为systemctl status docker.service

   查看images: docker images
   docker ps命令可以查看容器的CONTAINER ID、NAME、IMAGE NAME、端口开启及绑定、容器启动后执行的COMMNAD。最常用的功能是通过ps来找到CONTAINER_ID，以便对特定容器进行操作。
   docker ps 默认显示当前正在运行中的container
   docker ps -a 查看包括已经停止的所有容器
   docker ps -l 显示最新启动的一个容器（包括已停止的
   docker attach命令对应开发者很有用，可以连接到正在运行的容器，观察容器的运行状况，或与容器的主进程进行交互。
   ```


    从docker registry server 中下拉image或repository（pull）
    Usage: docker pull [OPTIONS] NAME[:TAG]
    eg： docker pull centos
    也可以明确指定具体的镜像：
    docker pull centos:centos6

    Docker环境信息 — docker [info|version]
    容器生命周期管理 — docker [create|exec|run|start|stop|restart|kill|rm|pause|unpause]
    容器操作运维 — docker [ps|inspect|top|attach|wait|export|port|rename|stat]
    容器rootfs命令 — docker [commit|cp|diff]
    镜像仓库 — docker [login|pull|push|search]
    本地镜像管理 — docker [build|images|rmi|tag|save|import|load]
    容器资源管理 — docker [volume|network]
    系统日志信息 — docker [events|history|logs]

```
2. Step 2. 安装nvidia-docker
[Link](https://www.jianshu.com/p/784d305a9d58)
```

curl https://get.docker.com | sh
sudo systemctl start docker && sudo systemctl enable docker

# 设置 stable 存储库和 GPG 密钥：

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list # 要访问 experimental 诸如 WSL 上的 CUDA 或 A100 上的新 MIG 功能之类的功能，您可能需要将 experimental 分支添加到存储库列表中.# # 可加可不加
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list # nvidia-docker2 更新软件包清单后，安装软件包（和依赖项）：
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd # 设置默认运行时后，重新启动 Docker 守护程序以完成安装：
sudo systemctl restart docker

    sudo docker run --runtime=nvidia --rm nvidia/cuda:11.1-base nvidia-smi

```
修改docker镜像保存数据路径 [LINK](https://blog.csdn.net/xiawenping/article/details/111461921)
(1) 首先停止 docker 服务：
systemctl stop docker
(2) 然后移动整个 /var/lib/docker 目录到目的路径(/data/docker)：
mv /var/lib/docker  /data/docker
(3) 创建软链接
ln -s /data/docker /var/lib/docker
Note：命令的意思是 /var/lib/docker 是链接文件名，其作用是当进入/var/lib/docker目录时，实际上是链接进入了 /data/docker 目录
(4) 重启 docker
systemctl start docker

```

# Docker 压缩

https://github.com/goldmann/docker-squash
