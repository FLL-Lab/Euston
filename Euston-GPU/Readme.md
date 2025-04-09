
# GPU Implementation Guide

## Requirements
- CUDA Toolkit = 12.2
- cuSOLVER library

## Build Instructions
```bash
mkdir build && cd build
cmake .. # Auto-detects PhantomFHE in thirdparty/
make -j4
```

## Executable Targets
```bash
build/bin/euston_main 
build/bin/nexus_main  
build/bin/bootstrapping 
```
