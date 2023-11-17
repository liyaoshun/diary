# **PAI平台使用**

## conda环境创建
conda create -p /datas/workspace/tool/dt_nerf python==3.8  # -p 表示安装路径
conda create -p /datas/workspaces/conda_envs/dt_nerf --clone dt_nerf  # 表示克隆路径
激活方式： conda activate /datas/workspaces/conda_envs/dt_nerf

conda init bash && bash
source activate
conda activate /datas/workspaces/conda_envs/dt_nerf


## mc 拷贝数据
1. 复制配置 并在cmd中运行
2. 复制示例，并在cmd中运行，找到需要下载文件得名称
3. mc cp -r  our-sdf-data/dataset-harix-our-sdf-data/our_sdf_data/qiyehao/qiyehao_jiehu_normal_0518_depth_normal.tar ./


## mc安装
下载minio-binaries
环境变量中添加: export PATH=$PATH:$HOME/minio-binaries/


