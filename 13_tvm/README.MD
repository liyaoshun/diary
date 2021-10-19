# **TVM 编译和部署pytorch模型记录**

[官方文档](https://tvm.apache.org/docs/install/from_source.html)

**环境要求**
```
1. A recent c++ compiler supporting C++ 14 (g++-5 or higher)
2. CMake 3.5 or higher
3. We highly recommend to build with LLVM to enable all the features.
4. If you want to use CUDA, CUDA toolkit version >= 8.0 is required. If you are upgrading from an older version, make sure you purge the older version and reboot after installation.
5. Python is also required. Avoid using Python 3.9.X+ which is not supported. 3.7.X+ and 3.8.X+ should be well supported however.
6. 
```

**LLVM编译**
```
LLVM 4.0 or higher is needed for build with LLVM. Note that version of LLVM from default apt may lower than 4.0.
```
Since LLVM takes long time to build from source, you can download pre-built version of LLVM from [LLVM Download Page.](http://releases.llvm.org/download.html)

1. 源码下载 git clone https://github.com/llvm/llvm-project.git

2. 然后根据readme.md进行编译



**TVM编译**

1. 下载tvm源码:  git clone --recursive https://github.com/apache/tvm tvm
2. 配置环境
```
1. sudo apt-get update
2. cd tvm && mkdir build
3. cp cmake/config.cmake build
4. 修改build吓得config.cmake文件，Change set(USE_CUDA OFF) to set(USE_CUDA ON) to enable CUDA backend. Do the same for other backends and libraries you want to build for (OpenCL, RCOM, METAL, VULKAN, …).     ||     To help with debugging, ensure the embedded graph executor and debugging functions are enabled with set(USE_GRAPH_EXECUTOR ON) and set(USE_PROFILER ON)    ||    To debug with IRs, set(USE_RELAY_DEBUG ON) and set environment variable TVM_LOG_DEBUG. export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"
5. set(USE_LLVM /path/to/your/llvm/bin/llvm-config)
6. cd build
7. cmake ..
8. make -j 12


```


