#ifndef CUDASHAREDPTR_H
#define CUDASHAREDPTR_H

//========================================================================
/*!
  @file
  @class
  @brief
    This header file provide smart shared pointer for device buffer.
    Cuda device pointer will be automatically destroyed when refcount reaches to zero.

  @author  Seyed Mahdi Roostaiyan, (C) 2020
*/
//========================================================================


#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace inner_implementation {
template <typename T_ELEM>
struct CudaPtrWrapper {
    size_t n_elements = 0;
    size_t alloc_size = 0;
    T_ELEM* data = nullptr;
    mutable cudaError_t status = cudaSuccess;
    bool buffer_reuse = false;

public:
    CudaPtrWrapper(bool buffer_pool_usage){
        buffer_reuse = buffer_pool_usage;
    }

    cudaError_t getStatus() const{
        return status;
    }
    bool create(size_t size){
        if(n_elements == size)
            return true;
        //! decrease number of cudaFree & cudaMalloc
        if(size<=alloc_size && buffer_reuse){
            n_elements = size;
            return true;
        }
        destroy();
        if(size<=0)
            return false;
        n_elements = size;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        status = cudaMalloc(&data, buffer_size);
        if(status!=cudaSuccess){
            std::cout << "Error: cudaMalloc was failed " << std::endl;
            return false;
        }
        alloc_size = n_elements;
        return true;
    }
    void upload(const T_ELEM* data_arr, size_t size, cudaStream_t stream){
        if(!create(size))
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        if(stream)
            status = cudaMemcpyAsync(data, data_arr, buffer_size, cudaMemcpyHostToDevice, stream);
        else
            status = cudaMemcpy(data, data_arr, buffer_size, cudaMemcpyHostToDevice);
        if(status!=cudaSuccess){
            std::cout << "Error: cudaMemcpy was failed " << std::endl;
            return;
        }
    }
    void download(T_ELEM* data_arr, cudaStream_t stream) const {
        if(n_elements<=0)
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        if(stream)
            status = cudaMemcpyAsync(data_arr, data, buffer_size, cudaMemcpyDeviceToHost, stream);
        else
            status = cudaMemcpy(data_arr, data, buffer_size, cudaMemcpyDeviceToHost);
        if(status!=cudaSuccess){
            std::cout << "Error: cudaMemcpy was failed " << std::endl;
            if(data)
                cudaFree(data);
            return;
        }
    }

    void clear(){
        if(data && !buffer_reuse){
            cudaFree(data);
            data = nullptr;
            alloc_size = 0;
        }
        n_elements = 0;
    }

    T_ELEM* pointer() const {
        if(n_elements>0)
            return data;
        else
            return nullptr;
    }

    void destroy(){
        if(data){
            cudaFree(data);
        }
        data = nullptr;
        n_elements = 0;
        alloc_size = 0;
    }

    ~CudaPtrWrapper(){
        destroy();
    }
};
}

namespace fun {

enum StreamPriority {
    Default = 0,
    Low  = 1,
    High = 2
};

}

namespace inner_implementation {
struct StreamPtrWrapper {
    cudaStream_t ptr = nullptr;

public:
    StreamPtrWrapper(fun::StreamPriority priority){
        if(priority==fun::Default){
            if (cudaStreamCreate(&ptr) != cudaSuccess)
                throw std::runtime_error("failed to create cuda stream");
        } else {
            int low, high;
            cudaDeviceGetStreamPriorityRange(&low, &high);
            int priority_num = (priority==fun::Low) ? low : high;
            if (cudaStreamCreateWithPriority(&ptr, cudaStreamNonBlocking, priority_num) != cudaSuccess)
                throw std::runtime_error("failed to create cuda stream");
        }
    }

    ~StreamPtrWrapper(){
        if(ptr)
            cudaStreamDestroy(ptr);
        ptr = nullptr;
    }

    void waitForCompletion() const {
        if(ptr)
            cudaStreamSynchronize(ptr);
     }
};
}

namespace fun {

namespace cuda {

template <typename T_ELEM>
struct shared_ptr {
private:
    std::shared_ptr<inner_implementation::CudaPtrWrapper<T_ELEM>> dev_array;

public:
    mutable std::shared_ptr<T_ELEM[]> host_arr;

public:
    shared_ptr(bool buffer_pool_usage=false){
        dev_array = std::make_shared<inner_implementation::CudaPtrWrapper<T_ELEM>>(buffer_pool_usage);
    }
    shared_ptr(size_t n_elements, bool buffer_pool_usage=false){
        dev_array = std::make_shared<inner_implementation::CudaPtrWrapper<T_ELEM>>(buffer_pool_usage);
        create(n_elements);
    }
    ~shared_ptr(){
        if(dev_array)
            dev_array.reset();
    }
    void destroy(){
        if(dev_array)
            dev_array->destroy();
        host_arr.reset();
    }
    void clear(){
        if(dev_array)
            dev_array->clear();
        host_arr.reset();
    }
    T_ELEM* data(){
        return dev_array->pointer();
    }

    const T_ELEM* data() const {
        return dev_array->pointer();
    }

    cudaError_t getStatus(){
        return dev_array->getStatus();
    }

    int size(){
        return dev_array->n_elements;
    }
    bool create(size_t n_elements){
        return dev_array->create(n_elements);
    }
    void upload_async(const std::vector<T_ELEM> &data_vec, cudaStream_t stream){
        size_t n_elements = data_vec.size();
        host_arr = std::shared_ptr<T_ELEM[]>(new T_ELEM[n_elements]);
        std::copy(data_vec.begin(), data_vec.end(), host_arr.get());
        dev_array->upload(host_arr.get(), n_elements, stream);
    }
    void upload(const std::vector<T_ELEM> &data_vec){
        upload_async(data_vec, nullptr);
    }
    void download_async(std::vector<T_ELEM> &data_vec, cudaStream_t stream) const {
        size_t n_elements = dev_array->n_elements;
        if(n_elements<=0)
            return;
        host_arr = std::shared_ptr<T_ELEM[]>(new T_ELEM[n_elements]);
        dev_array->download(host_arr.get(), stream);
        if(stream)
            cudaStreamSynchronize(stream);
        std::copy(host_arr.get(), host_arr.get()+n_elements, data_vec.begin());
    }
    void download(std::vector<T_ELEM> &data_vec) const {
        download_async(data_vec, nullptr);
    }
    void upload_async(const T_ELEM* data_arr, size_t n_elements, cudaStream_t stream){
        dev_array->upload(data_arr, n_elements, stream);
    }
    void upload(const T_ELEM* data_arr, size_t n_elements){
        upload_async(data_arr, n_elements, nullptr);
    }
    void download_async(T_ELEM* data_vec, cudaStream_t stream) const {
        dev_array->download(data_vec, stream);
    }
    void download(T_ELEM* data_vec) const {
        download_async(data_vec, nullptr);
    }
};


struct Stream {
    std::shared_ptr<inner_implementation::StreamPtrWrapper> c_stream;
    Stream(StreamPriority priority) {
        c_stream = std::make_shared<inner_implementation::StreamPtrWrapper>(priority);
    }
    ~Stream(){
        c_stream.reset();
    }
    void waitForCompletion() const {
        if(c_stream)
            c_stream->waitForCompletion();
    }
};

typedef  std::shared_ptr<fun::cuda::Stream> PriorityStreamPtr;
}
}
#endif // CUDASHAREDPTR_H

