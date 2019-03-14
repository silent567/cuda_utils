# cuda_utils
Wrapper of C++ CUDA functions for accessing GPU information in python

Utilize the CUDA functions compiled in libcuda.so . 
Two functions are implemented: 
- GPU_count: return the number of visible GPU devices
- GPU_with_max_mem (only support 32 bit memory-location i.e. <= 4GB): return the index of GPU with maximum available memory. 

The functions are originally implemented for assigning GPUs dynamically especially in the multiprocessing situations. Early stopping makes the computation time of different processes different. And I want to assign GPU to process's tf session adaptively by choosing the one with maximum free GPU memory. 
But latter, I found that multiprocessing.Pool reuses processes instead of creating new ones. So, the better solution is to simply combine GPU number with the process_id. A detailed description is available in this [Chinese blog](
python获取GPU相关信息).

The second function can not distinguish free GPU memory bigger than 4GB maybe because it utilizes int32 for pointers inside the CUDA library. 
Other possible solutions for accessing GPU information are discussed in this [Chinese blog](https://blog.csdn.net/silent56_th/article/details/81320067).
