# CPU Implementation Guide

## Dependencies

### Eigen3 (v3.4 Strict)
```bash
sudo apt install libeigen3-dev
# Ensure header accessibility
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include
```

### Modified SEAL (v4.1-bs)
```bash
cd thirdparty/SEAL-4.1-bs
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=../../SEALlibs
cmake --build build
sudo cmake --install build
```

## Build Process
```bash
mkdir build && cd build
cmake .. 
make -j4
```

## Executables
```bash
build/bin/euston_main    # Main inference engine
build/bin/nexus_main     # Secure compression module
build/bin/bootstrapping  # Depth optimization
```
