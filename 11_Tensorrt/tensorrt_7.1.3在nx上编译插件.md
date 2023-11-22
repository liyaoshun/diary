# **tensorrt_7.1.3在nx上编译插件**

## 编译命令
cmake .. -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2 -DGPU_ARCHS=72 -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++