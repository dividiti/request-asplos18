# Classification program for ArmCL

ImageNet classification and benchmarking using ArmCL and MobileNet.

## Requirements

### ArmCL library
To build this program, you need ArmCL compiled with Graph API:
```
ck install package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph
```

To build this program for Android you need to embedd kernels and select target API as follows:
```
$ ck install package:lib-armcl-opencl-18.05 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph --env.USE_EMBEDDED_KERNELS=ON --env.DEBUG=ON --target_os=android23-arm64
```

**NB:** Use `--target_os=android23-arm64` to build for Android API 23 (v6.0 "Marshmallow") or [similar](https://source.android.com/setup/start/build-numbers).

We have to embed kernels when building for Android as OpenCL kernel files are not copied to a remote device.

**TODO:** For some reason only debug version of the library can be used with this program on Android. When we use release version, the program gets stuck at graph preparation stage.

### Weights package
One of MobileNet weights packages:
```
ck install package:weights-mobilenet-v1-1.0-224-npy
ck install package:weights-mobilenet-v1-0.75-224-npy
ck install package:weights-mobilenet-v1-0.50-224-npy
ck install package:weights-mobilenet-v1-0.25-224-npy

ck install package:weights-mobilenet-v1-1.0-192-npy
ck install package:weights-mobilenet-v1-0.75-192-npy
ck install package:weights-mobilenet-v1-0.50-192-npy
ck install package:weights-mobilenet-v1-0.25-192-npy

ck install package:weights-mobilenet-v1-1.0-160-npy
ck install package:weights-mobilenet-v1-0.75-160-npy
ck install package:weights-mobilenet-v1-0.50-160-npy
ck install package:weights-mobilenet-v1-0.25-160-npy

ck install package:weights-mobilenet-v1-1.0-128-npy
ck install package:weights-mobilenet-v1-0.75-128-npy
ck install package:weights-mobilenet-v1-0.50-128-npy
ck install package:weights-mobilenet-v1-0.25-128-npy
```

### ImageNet dataset
```
ck install package:imagenet-2012-val
ck install package:imagenet-2012-aux
```

## Build
```
ck compile program:mobilenet-armcl-opencl [--target_os=android23-arm64] 
```

## Run
```
ck run program:mobilenet-armcl-opencl [--target_os=android23-arm64] 
```
