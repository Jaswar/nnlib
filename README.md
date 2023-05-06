## Overview

*nnlib* is a GPU-accelerated, static, C/C++ neural network library. It was designed to work in one of the following two modes:
 - CPU-only: All operations take place on CPU and all data is stored in the main memory. Single Instruction Multiple Data (SIMD) 
instruction sets, AVX and AVX2, are used to increase performance.
 - GPU-accelerated: Most of the operations are performed on GPU and most of the data is stored on GPU' memory. The library then
uses the GPU to increase performance in parallelizable tasks such as matrix multiplication. An Nvidia, CUDA-capable GPU is required 
to run the library in this mode.

### Supported functionalities

 - Layers:
   - Fully connected
 - Activation functions:
   - Linear
   - ReLU
   - Sigmoid
 - Loss functions:
   - Mean Squared Error
   - Binary Cross Entropy
   - Categorical Cross Entropy
 - Metrics:
   - All of the above loss functions
   - Categorical accuracy
   - Binary accuracy
 - Optimizers:
   - Stochastic Gradient Descent

## Setup 

The library is currently supported on both Linux and Windows.

### Linux

1. Install a C++ compiler. Both `g++` and `clang++` have been tested.
```shell
# For g++
sudo apt install g++
# For clang++
sudo apt install clang  
```
2. Install [CMake](https://cmake.org/) version at least 3.16. Can be downloaded from the official site: https://cmake.org/download/.
3. If you do not wish to use GPU acceleration, go to step 4. Otherwise, continue with the following:
   1. Make sure that you have a [CUDA-Capable Nvidia GPU](https://en.wikipedia.org/wiki/CUDA). Only those GPUs are supported by the library.
   2. Install the NVIDIA CUDA Toolkit using the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
   3. Verify that CUDA was installed by running `nvcc -V` in the terminal.
4. Clone the repository using the following command:
```shell
git clone https://github.com/Jaswar/nnlib.git
```
5. Build and install the library. Within the cloned git folder do the following:
```shell
cd scripts
sudo chmod +x build.sh
./build.sh 
```
This will install the library in the `./install` directory in the main directory of the cloned repository (so **not** inside `scripts`).

### Windows

The following is the recommended way to set up the library on Windows. The Visual Studio
setup is required for the installation of CUDA and running the library in GPU-accelerated mode. Visual Studio is not
however required to run the library in CPU-only mode. For that, a C++ compiler such as Clang/GCC will be sufficient.

1. Download and install Visual Studio 2017 or higher.
2. Enable *Desktop development with C++* in the Visual Studio Installer.
3. If you do not wish to use GPU-acceleration, go to step 4. Otherwise, continue with the steps below:
   1. Make sure that you have a [CUDA-Capable Nvidia GPU](https://en.wikipedia.org/wiki/CUDA). Only those GPUs are supported by the library.
   2. Install the NVIDIA CUDA Toolkit using the [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
   3. Verify that CUDA was installed by running `nvcc -V` in the terminal.
4. Clone the repository using the following command:
```shell
git clone https://github.com/Jaswar/nnlib.git
```
5. It is recommended to build the library and the examples using the provided scripts in the `scripts` directory. [Git Bash](https://gitforwindows.org/) can
be used for that purpose. If you do not have Git Bash installed, an alternative could be a tool such as [Cygwin](https://www.cygwin.com/) or [MinGW](https://www.mingw-w64.org/).

The library can then be installed using the following commands (inside the cloned repository):
```shell
cd scripts
./build.sh
```
This will install the library in the `install` folder in the main directory.

## Running an example

1. Install the library as described in the Setup.
2. Build an example. Here we will build the MNIST example (assuming we start in the main directory):
```shell
cd scripts

# If you are on Linux give execute permissions to the file
sudo chmod +x build_example.sh

./build_example.sh -c Release -p <path_to_nnlib_install> mnist
```
Here, `<path_to_nnlib_install>` should be replaced with the **absolute** path to the `install` folder
that was created in the library installation step. Running these commands will build the mnist example in `./examples/mnist/build`.
3. Run the example. It expects the **absolute** path to a `MNIST_train.txt` file, which can be downloaded from [here](https://github.com/halimb/MNIST-txt),
   as the first and only argument. Run the example with the following command:
```shell
cd examples/mnist/build
./mnist_nnlib <path_to_MNIST_train.txt>
```

## Documentation

The whole project is documented on https://jaswar.github.io/nnlib.
