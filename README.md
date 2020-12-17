# CudaSharedPtr

This header file provides a smart pointer of device memory (cuda::shared_ptr<T>) , 
which is released automatically whenever is needed exactly similar to std::shared_ptr.

To upload and download the array of host objects into the Cuda device you can use below functions 
(To upload a single object, you can set n_elements = 1):

    void upload_async(const std::vector<T> &data_vec, cudaStream_t stream);
    void upload(const std::vector<T> &data_vec);
    void download_async(std::vector<T> &data_vec, cudaStream_t stream) const;
    void download(std::vector<T> &data_vec) const;
    void upload_async(const T* data_arr, int n_elements, cudaStream_t stream);
    void upload(const T* data_arr, int n_elements);
    void download_async(T* data_vec, cudaStream_t stream) const;
    void download(T_ELEM* data_vec) const;
    
Usage Example:


    std::shared_ptr<T[]> data_host = = std::shared_ptr<T[]>(new T[n]);
    .
    .
    .

    // In host code:
    cuda::shared_ptr<T> data_dev;
    data_dev->upload(data_host.get(), n);
    // In .cu file:
    // data_dev.data() points to device memory which contains data_host;

