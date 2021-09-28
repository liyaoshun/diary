#  **ROS相关问题**

## 1. 在安装ROS的时候(运行命令：sudo apt install ros-melodic-desktop-full时发生错误。)
   
```
ros-melodic-desktop-full : Depends: ros-melodic-simulators but it is not going to be installed



然后也发现在本机上的python2.7.*版本比本机的要高。同时python-dev也没有安装。
python-dev : Depends: python (= 2.7.15~rc1-1) but 2.7.16-1 is to be installed

解决方案：

1. sudo aptitude install python-dev  会显示python-dev [未安装的]  
2. 是否接受该解决方案？[Y/n/q/?] n
/** 显示降级操作
3. 是否接受该解决方案？[Y/n/q/?] y
4. 您要继续吗？[Y/n/?]


然后重新执行：sudo apt install ros-melodic-desktop-full

```

## 2. Gtk-ERROR **: 09:57:47.834: GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported

