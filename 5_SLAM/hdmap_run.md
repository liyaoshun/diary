# **HDMap 使用手册**

```
run docker command:
ROBOMAP=$(pwd)

if [ $# -ne 2 ]; then
    CONTAINER=gingerlite_cd
else
    CONTAINER=$2
fi

if [ $# -lt 1 ] || [ "$1" != "1" ]; then
    docker container stop $CONTAINER
    docker container rm $CONTAINER
    docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --name $CONTAINER\
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --privileged=true \
        --security-opt seccomp=unconfined \
        --hostname=$CONTAINER \
        -v /etc/localtime:/etc/localtime:ro \
        -v /etc/timezone:/etc/timezone:ro \
        -v "${ROBOMAP}:/robomap" \
	    -v "/media/robot/nvme2T/rosbag/tanggong/:/rosdata" \
        -v "/:/linux" $CONTAINER /bin/bash && roscore
elif [ "$1" -eq "1" ]; then
    docker exec -it -e NVIDIA_VISIBLE_DEVICES=0 \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged=true \
        $CONTAINER /bin/bash && roscore
fi
```

## **建图相关配置和命令**

### **建图配置 (Step 1)**

**建图配置**

```
hd_map_mapping.launch
1. log_dir ： 输出日志路径
2. output_folder ： 建图输出结果
3. is_simulation ： 使用bag建图时需要设置为true

mapping.yaml
1. semantic_points_model : 关键点、描述子提取模型路径(.trt) | superpoint
2. semantic_match_model : superglue模型路径
3. use_spglue ： 设置为true的时候表示使用神经网络提取特征点
```

**建图使用命令**

编译

```
catkin_build.sh

export ROS_VERSION=melodic
catkin init
catkin config --install
catkin config --merge-devel
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build

```

建图

```
1. source devel/setup.bash # 切换到hd_map工作空间
2. roslaunch hd_map_node hd_map_mapping.launch # 开始建图
3. rosplay comm: rosbag play bag_path --clock
4. 完成后输出enter开始保存建图信息
```

### **合图配置 (Step 2)**

```
map_merge.launch
1. log_dir : 合图日志文件保存路径
2. input_map_folder : 上一步mapping的输出路径
3. output_folder ： 合图输出路径
4. use_lidar2d ： 合图需要设置为true，当前合图依赖2D激光


merge.yaml
1. enable_visualization ： 可视化合图开关，可以设置为true来检查合图效果
```

```
roslaunch map_merge_node map_merge.launch input_map_folder:=/rosdata/ros/test output_folder:=/rosdata/ros/merge_rst use_vision:=true use_lidar2d:=true
```

### **定位配置 (Step 3)**

```
hd_map_localization.launch
1. inputmap_path : 合图输出路径，并且需要指定到 sparse_pc_layer 文件夹
2. log_dir ： 定位日志输出路径
3. --v : 输出日志的级别

localization.yaml
semantic_points_model : 关键点、描述子提取模型路径(.trt) | superpoint
semantic_match_model ： superglue模型路径
use_spglue ： 控制使用神经网络提取特征进行定位


```

## **可视化命令**

```
1. 在docker里面运行roscore &
2. 在本机上运行 xhost +
2. 在docker里面运行rviz打开可视化界面
```

## **其他命令**

```
同一个docker打开多个窗口命令：  gingerlite_cd：image
docker exec -it gingerlite_cd /bin/bash
```
