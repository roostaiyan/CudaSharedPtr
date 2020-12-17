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
    size_t n_elements;
    T_ELEM* data = nullptr;

public:
    void upload(const T_ELEM* data_arr, size_t size, cudaStream_t stream){
        destroy();
        n_elements = size;
        if(n_elements<=0)
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        cudaError_t status = cudaMalloc(&data, buffer_size);
        if(status!=cudaSuccess){
            std::cout << "Error: cudaMalloc was failed " << std::endl;
            return;
        }
        if(stream)
            status = cudaMemcpyAsync(data, data_arr, buffer_size, cudaMemcpyHostToDevice, stream);
        else
            status = cudaMemcpy(data, data_arr, buffer_size, cudaMemcpyHostToDevice);
        if(status!=cudaSuccess){
            std::cout << "Error: cudaMemcpy was failed " << std::endl;
            if(data)
                cudaFree(data);
            return;
        }
    }
    void download(T_ELEM* data_arr, cudaStream_t stream) const {
        if(n_elements<=0)
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        cudaError_t status;
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
    void destroy(){
        if(data){
            cudaFree(data);
        }
        data = nullptr;
        n_elements = 0;
    }
    ~CudaPtrWrapper(){
        destroy();
    }
};
}

namespace cuda {

template <typename T_ELEM>
struct shared_ptr {
private:
    std::shared_ptr<inner_implementation::CudaPtrWrapper<T_ELEM>> dev_array;
public:
    shared_ptr(){
        dev_array = std::make_shared<inner_implementation::CudaPtrWrapper<T_ELEM>>();
    }
    ~shared_ptr(){
        if(dev_array)
            dev_array.reset();
    }
    T_ELEM* data(){
        return dev_array->data;
    }
    size_t size(){
        return dev_array->n_elements;
    }
    void upload_async(const std::vector<T_ELEM> &data_vec, cudaStream_t stream){
        int n_elements = data_vec.size();
        std::unique_ptr<T_ELEM[]> data_arr = std::unique_ptr<T_ELEM[]>(new T_ELEM[n_elements]);
        std::copy(data_vec.begin(), data_vec.end(), data_arr.get());
        dev_array->upload(data_arr.get(), n_elements, stream);
    }
    void upload(const std::vector<T_ELEM> &data_vec){
        upload_async(data_vec, nullptr);
    }
    void download_async(std::vector<T_ELEM> &data_vec, cudaStream_t stream) const {
        int n_elements = dev_array->n_elements;
        if(n_elements<=0)
            return;
        std::unique_ptr<T_ELEM[]> data_arr = std::unique_ptr<T_ELEM[]>(new T_ELEM[n_elements]);
        dev_array->download(data_arr.get(), stream);
        std::copy(data_arr.get(), data_arr.get()+n_elements, data_vec.begin());
    }
    void download(std::vector<T_ELEM> &data_vec) const {
        download_async(data_vec, nullptr);
    }
    void upload_async(const T_ELEM* data_arr, int n_elements, cudaStream_t stream){
        dev_array->upload(data_arr, n_elements, stream);
    }
    void upload(const T_ELEM* data_arr, int n_elements){
        upload_async(data_arr, n_elements, nullptr);
    }
    void download_async(T_ELEM* data_vec, cudaStream_t stream) const {
        dev_array->download(data_vec, stream);
    }
    void download(T_ELEM* data_vec) const {
        download_async(data_vec, nullptr);
    }
};
}

#endif // CUDASHAREDPTR_H

