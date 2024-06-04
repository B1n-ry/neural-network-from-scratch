# Hand written digits detection
I wrote this application to learn about how neural networks learn on a fundamental level,
and therefore does not use libraries to simplify the learning or anything the network does.
This project also lead me to learn about GPU utilization with CUDA, as training a neural
network completely on the CPU would simply take too long.

This is a neural network built to recognize hand written digits, based on a youtube-series
by 3Blue1Brown: [Neural Networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=BhZKhQbyJDIU6oUW)

As I'm trying to learn rust simultaneously as learning neural networks, this network
is written in rust, with the help of the crate `cudarc` to utilize the GPU.
For this I am also using CUDA to write code that will run on the GPU