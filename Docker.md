## 拷贝宿主机文件到container
nvidia-docker cp  /**/**/demo.txt  mycontainer:/sss/sss/
提交修改，保证下次打开修改了的文件还存在。
sudo docker commit -m "描述内容" -a "author name" 容器id images_name:tag