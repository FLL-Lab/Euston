要在CPU上编译Euston，首先需要安装Eigen和SEAL两个依赖库。
# 安装Eigen
Eigen的版本被限制在了Eigen3.4。
可以通过如下方式进行安装：

sudo apt install libeigen3-dev
 
//若默认安装的是/usr/local/include/eigen3/Eigen 下，将Eigen文件夹拷贝一份到/usr/local/include 下，这样便于直接include <Eigen>头文件
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include

# 安装SEAL
我们使用了NEXUS修改后的SEAL4.1版本，这是被放在thirdparty/SEAL-4.1-bs目录下的。
SEAL的安装方式可以参考原项目的Readme。
我们建议install Microsoft SEAL locally, e.g., to `Euston-CPU/SEALlibs/`,这样可以避免影响系统环境的SEAL库:

```PowerShell
cd thirdparty/SEAL-4.1-bs
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../../SEALlibs
cmake --build build
sudo cmake --install build
```

# 编译Euston和NEXUS
在安装完依赖后，我们可以直接通过如下方式完成项目的构建。
```PowerShell
mkdir build && cd build
cmake ..
make 
```

# 运行Euston和NEXUS
我们可以在build/bin下找到对应的执行文件。
bin/euston_main
bin/nexus_main
bin/bootstrapping