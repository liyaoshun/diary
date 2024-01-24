
#................docker 使用相关命令 .....................#  https://www.cnblogs.com/duanxz/p/7905233.html

查看images: docker images

docker ps 命令可以查看容器的CONTAINER ID、NAME、IMAGE NAME、端口开启及绑定、容器启动后执行的COMMNAD。最常用的功能是通过ps来找到CONTAINER_ID，以便对特定容器进行操作。
docker ps 默认显示当前正在运行中的container
docker ps -a 查看包括已经停止的所有容器
docker ps -l 显示最新启动的一个容器（包括已停止的

sudo docker container prune  停止容器


docker attach命令对应开发者很有用，可以连接到正在运行的容器，观察容器的运行状况，或与容器的主进程进行交互。

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


#................docker 使用相关命令 .....................#  https://www.cnblogs.com/duanxz/p/7905233.html
查看images: docker images

docker ps命令可以查看容器的CONTAINER ID、NAME、IMAGE NAME、端口开启及绑定、容器启动后执行的COMMNAD。最常用的功能是通过ps来找到CONTAINER_ID，以便对特定容器进行操作。
docker ps 默认显示当前正在运行中的container
docker ps -a 查看包括已经停止的所有容器
docker ps -l 显示最新启动的一个容器（包括已停止的

$ docker ps // 查看所有正在运行容器
$ docker stop containerId // containerId 是容器的ID

$ docker ps -a // 查看所有容器
$ docker ps -a -q // 查看所有容器ID

$ docker stop $(docker ps -a -q) //  stop停止所有容器
$ docker  rm $(docker ps -a -q) //   remove删除所有容器


docker --version //查看version
docker images //查看所有镜像
docker rm [NAME]/[CONTAINER ID] //删除容器 :不能够删除一个正在运行的容器，会报错。需要先停止容器

sudo docker container prune  // 停止容器 删除所有关闭的容器


docker attach命令对应开发者很有用，可以连接到正在运行的容器，观察容器的运行状况，或与容器的主进程进行交互。

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

容器管理
docker run -i -t <IMAGE_ID> /bin/bash：-i：标准输入给容器 -t：分配一个虚拟终端 /bin/bash：执行bash脚本
-d：以守护进程方式运行（后台）
-P：默认匹配docker容器的5000端口号到宿主机的49153 to 65535端口
-p <HOT_PORT>:<CONTAINER_PORT>：指定端口号

-name： 指定容器的名称
-rm：退出时删除容器
docker stop <CONTAINER_ID>： 停止container
docker start <CONTAINER_ID> ： 重新启动container
docker ps - Lists containers.
-l：显示最后启动的容器
-a：同时显示停止的容器，默认只显示启动状态

docker attach <CONTAINER_ID> 连接到启动的容器
docker logs <CONTAINER_ID> : 输出容器日志
-f：实时输出
docker cp <CONTAINER_ID>:path hostpath：复制容器内的文件到宿主机目录上
docker rm <CONTAINER_ID>：删除container
docker rm docker ps -a -q：删除所有容器
docker kill docker ps -q
docker rmi docker images -q -a
docker wait <CONTAINER_ID>：阻塞对容器的其他调用方法，直到容器停止后退出

docker top <CONTAINER_ID>：查看容器中运行的进程
docker diff <CONTAINER_ID>：查看容器中的变化
docker inspect <CONTAINER_ID>：查看容器详细信息（输出为Json）
-f：查找特定信息，如 docker inspect - f ‘{{ .NetworkSettings.IPAddress }}’
docker commit -m “comment” -a “author” <CONTAINER_ID> ouruser/imagename:tag


## docker打开多个界面：
docker exec -it 1f910685ef86 /bin/bash   # 1f910685ef86:容器id  或者使用 image name

## docker 中使用rviz

## docker 存在容器后打开命令
docker start container_id
