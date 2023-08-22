tiny-cuda-nn 1.7 需要安装gcc9

首先，添加ubuntu-toolchain-r/test PPA 软件源到你的系统：
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

安装想要安装的 GCC 和 G++版本，输入：
sudo apt install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9


sudo apt install gcc-10 g++-10
下面的命令将会配置 alternative,并且设置优先级。默认的版本是最高优先级的一个，在我们的例子中是gcc-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7 --slave /usr/bin/gcov gcov /usr/bin/gcov-7

后面，如果你想更改你的默认版本，你可以使用update-alternatives命令：
sudo update-alternatives --config gcc

There are 3 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-9   90        auto mode
  1            /usr/bin/gcc-7   70        manual mode
  2            /usr/bin/gcc-8   80        manual mode
  3            /usr/bin/gcc-9   90        manual mode

Press <enter> to keep the current choice[*], or type selection number:
应该会有一列 系统上的 GCC 版本展示在你的面前。输入你想设置成默认版本的数字，按Enter回车键。
这个命令将会创建一个虚拟链接，指向指定版本的 GCC 和 G++