
# 内存使用检查

## 工具 

```
1. valgrind

安装和使用:
sudo apt-get install valgrind
valgrind --tool=memcheck --leak-check=full  ./Demo  # 运行当前目录下的Demo程序，检查Demo程序是否有内存泄露


sudo apt install massif-visualizer  可视化内存使用情况

Massif Visualizer is a graphical interface to visualize memory usage recorded by Valgrind Massif tool. . Run your application in Valgrind with --tool=massif and then open the generated massif.out.%pid in the visualizer. Gzip or Bzip2 compressed Massif files can also be opened transparently.
```