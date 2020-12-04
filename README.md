# Description
This project is a comparison between use warp-level primitives instead of shared memory. In this case the project is comparing a reduction with shared
memory classical approach vs warp-level functions.

# Building
The project has a CMakeLists.txt file to be built with cmake so:

```bash
$ mkdir build
$ cd build
$ cmake ../
$ make
```

# TODO
* Add the warp-level variant
* Use shfl_down or reduce_sum warp-leve instruction depending on __CUDA_ARCH__
