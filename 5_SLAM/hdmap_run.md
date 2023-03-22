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
ginger_lite_mapping.launch
1. log_dir ： 输出日志路径
2. output_folder ： 建图输出结果
3. is_simulation ： 使用bag建图时需要设置为true
4. vision_subtype ： 设置为true的时候表示使用神经网络提取特征点
5. camera_type： 设置相机类型

mapping.yaml
1. semantic_points_model : 关键点、描述子提取模型路径(.trt) | superpoint
2. semantic_match_model : superglue模型路径

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
2. roslaunch hd_map_node ginger_mapping.launch # 开始建图
3. rosplay comm: rosbag play bag_path --clock
4. 完成后输出enter开始保存建图信息
5. rviz 开启可视化
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

 roslaunch map_merge_node map_merge.launch input_map_folder:=/rosdata/ros/mapping_map/vi-maps output_folder:=/rosdata/ros/merge_map use_vision:=true use_lidar2d:=true
```

### **定位配置 (Step 3)**

```
hd_map_localization.launch
1. inputmap_path : 合图输出路径，并且需要指定到 sparse_pc_layer 文件夹
2. log_dir ： 定位日志输出路径
3. --v : 输出日志的级别
4.camera_type
5.vision_subtype

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


## **字典训练-使用合图后数据**
```
rosrun hd_map_console hd_map_console
load --map_folder /data/user/ginger/liys/mapping_out/merge_map/result_for_next_merge

tri_opt_all # 三角化map

tm --lc_number_of_vocabulary_words 100 --train_unstable_num 500 --lc_projection_matrix_filename /data/user/ginger/liys/hdmap/src/hd_map/algorithms/loopclosure/matching_based_loopclosure/share/projection_matrix_spglue.dat --lc_projected_quantizer_filename /data/user/ginger/liys/dic/fisheye.dat
lc_number_of_vocabulary_words : 表示100*100个类别    train_unstable_num：结束标志。 lc_projection_matrix_filename：投影矩阵，当前未使用。指向仓库中已有的。    lc_projected_quantizer_filename： 新分类文件。

修改分类文件路径：/data/user/ginger/liys/hdmap/src/hd_map/algorithms/loopclosure/matching_based_loopclosure/src/detector-settings.cc       
projected_quantizer_filename = std::string(loop_closure_files_path) + "/inverted_multi_index_quantizer_supoint.dat";

```

## **字典训练-未合图数据**
```
在进行第一次fp训练分类文件的时候需要将mapping-workflows-plugin.cc中79、80行代码注释掉。
      // mapping_workflows_plugin::processVIMapToLocalizationMap(
      //     kInitializeLandmarks, keyframe_options, map.get(), plotter);
等训练好了分类文件再打开重新编译一下
```


## **评价定位**
```
1.定位阶段需要将 --v 3
2.在hdmap_tools中使用脚本 /media/robot/nvme2T/docker/hdmap_tools/analysis_loc/analysisLoc.py   # python 3.8 以上 plenoxel
eg:python analysisLoc.py -i /media/robot/nvme2T/rosbag/log/hd_map_node.INFO
```