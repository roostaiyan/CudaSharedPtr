# CudaSharedPtr (cuda::shared_ptr)

This header file provides a smart pointer of device memory (cuda::shared_ptr<T>) , 
which is released automatically whenever is needed exactly similar to std::shared_ptr (when ref_count reaches to zero). You can also use the same technique for OpenCl Buffer and OpenGl Textures (texture id). 

To upload and download the array of host objects into the Cuda device you can use below functions 
(To upload a single object, you can set n_elements = 1):
    
    bool create(size_t size);//! you can use create to allocate memory or simply upload your data without calling this function
    void upload_async(const std::vector<T> &data_vec, cudaStream_t stream);
    void upload(const std::vector<T> &data_vec);
    void download_async(std::vector<T> &data_vec, cudaStream_t stream) const;
    void download(std::vector<T> &data_vec) const;
    void upload_async(const T* data_arr, int n_elements, cudaStream_t stream);
    void upload(const T* data_arr, int n_elements);
    void download_async(T* data_vec, cudaStream_t stream) const;
    void download(T_ELEM* data_vec) const;
    
Usage Example:


    std::shared_ptr<T[]> data_host = std::shared_ptr<T[]>(new T[n]);
    .
    .
    .

    // In host code:
    fun::cuda::shared_ptr<T> data_dev;
    data_dev->upload(data_host.get(), n);
    // In .cu file:
    // data_dev.data() points to device memory which contains data_host;

To decrease the number of cudaMalloc and cudaFree calls, you can pass buffer_pool_usage = true to fun::cuda::shared_ptr<T> constructor. This (buffer_pool_usage = true) is specially usefull when the maximum buffer length is limited and does not change after a while. This argument is "false" by default. 
