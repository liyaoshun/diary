
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