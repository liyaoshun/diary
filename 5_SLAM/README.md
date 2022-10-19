1. 如果使用语义分割的方法对每一帧都进行语义分割，那么建立关键点的时候就可以添加一个类别 id 信息，在构建数据关键目标函数的时候加入该信息的误差(那三篇中有一篇的思路就是这样)。
2. 如果是用目标检测来对每一帧进行目标画框的话，有一些目标框会重叠，我想可能还需要根据点云聚类分割的结果来给像素点打标签，然后采用上面的思路，不过有些点有标签，有些又没有，这样的匹配不知道要如何处理，或许就直接剔除给匹配点的语义误差。

## **rosbag play 播放数据包**

```
1. rosbag play recorded1.bag recorded2.bag

如果播放两个及以上bag包，那么他们会第一帧对齐，后面根据第一帧时间戳的时间差播放。

2. rosbag play -s 5 recorded1.bag

表示从bag的第几s开始播放

3. rosbag play -l recorded1.bag

表示循环播放此bag包

4. rosbag play -r 10 recorded1.bag

表示快进10倍播放此bag包，以录制频率的10倍回放。

5. rosbag play --pause record.bag

表示以暂停的方式启动，防止跑掉数据

6. rosbag play -u 10 record.bag

表示播放前10s的数据
```

docker exec -it gingerlite_cd /bin/bash
