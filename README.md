# Graph Convolutional Network (GCN)
This is an implementation of a GCN in C++ with BLIS and rsblib for matrix multiplication.

## Build
The included libraries have been compiled for M1 Macs and should be rebuild for different targets if run on different architectures.
It's recommended to enable multithreading when configuring BLIS for better performance.

The included compiled library versions are:
- BLIS (commit c6546c1)
- rsblib (1.3)
