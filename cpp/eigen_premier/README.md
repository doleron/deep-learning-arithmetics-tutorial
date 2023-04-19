# Eigen Premier
This is a C++ simple project to make you start up using Eigen.

## Hidding Eigen
The Eigen types names are hidden in `matrix_defininitions.hpp` using Type Alias:

```c++
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
```

## Building and running

To build the project:

```bash
$ cd eigen_premier
eigen_premier$ mkdir build
eigen_premier$ cd build
eigen_premier/build$ cmake ..
eigen_premier/build$ make
```

To run:

```bash
eigen_premier/build$ ./eigen_premier
```

The program should outputs something like:

```
eigen_premier/build$ ./eigen_premier 
1 eigen threads
t-1     0       81.1688 fps     1232 ms
t-2     0       80.5153 fps     1242 ms
t-0     0       80.3858 fps     1244 ms
t-1     1       80.6452 fps     1240 ms
t-2     1       80.5153 fps     1242 ms
t-0     1       80.3858 fps     1244 ms
t-1     2       81.3008 fps     1230 ms
t-2     2       81.0373 fps     1234 ms
t-0     2       80.7754 fps     1238 ms
...
```