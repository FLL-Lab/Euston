# 安装依赖
Euston依赖于CUDA::cusolver库，因此需要先安装cuda-toolkit。

# 编译Euston和NEXUS
在GPU上构建时我们可以直接通过如下命令编译：
```PowerShell
mkdir build && cd build
cmake ..
make 
```
这会利用thirdparty中的PhantomFHE完成构建。

# 运行Euston和NEXUS
我们可以在build/bin下找到对应的执行文件。
bin/euston_main
bin/nexus_main
bin/bootstrapping