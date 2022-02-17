/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GatherElement.h"
#include "cuda_fp16.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include "half.h"
#include <cstring>

using namespace nvinfer1;
using namespace plugin;


__global__ void GatherElementKernel(const void* data, 
                                    const void* index,
                                    float* output,
                                    int*  index_d,
                                    int index_nbDims,
                                    int temp_Ptr [],
                                    int*  data_d,
                                    int  N,
                                    int axis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)   // 判断是否超限
        return;

    int dev_Ptr[] = {temp_Ptr[0], temp_Ptr[1], temp_Ptr[2]};

    for(int i = 0; i<index_nbDims; ++i)
    {
        int temp = 1;
        for(int j=i+1; j<index_nbDims; ++j)
        {
            temp =  temp * index_d[j];
        }

        if (i>=1 && i!=index_nbDims-1)
        {
            dev_Ptr[i] = (int(idx/temp))%index_d[i];//(int(idx/temp))%index_d[i];
            // printf("ssss, idx=%d, dev_Ptr[0]=%d, dev_Ptr[1]=%d, dev_Ptr[2]=%d, dev_Ptr[i]=%d, i = %d, index_d[i]=%d， calc_value=%d, temp=%d \n", idx, dev_Ptr[0], dev_Ptr[1], dev_Ptr[2], dev_Ptr[i], i, index_d[i], (int(idx/temp))%index_d[i], temp);
        }else if(i==index_nbDims-1){
            dev_Ptr[i] = int(idx%index_d[i]);
            // printf("xxxx, idx=%d, dev_Ptr[0]=%d, dev_Ptr[1]=%d, dev_Ptr[2]=%d, dev_Ptr[i]=%d, i = %d, index_d[i]=%d, calc_value=%d \n", idx, dev_Ptr[0], dev_Ptr[1], dev_Ptr[2], dev_Ptr[i], i, index_d[i], idx%index_d[i]);
        }else{
            dev_Ptr[i] = int(idx/temp);
            // printf("cccc, idx=%d, dev_Ptr[0]=%d, dev_Ptr[1]=%d, dev_Ptr[2]=%d, dev_Ptr[i]=%d, i = %d, index_d[i]=%d, calc_value=%d \n", idx, dev_Ptr[0], dev_Ptr[1], dev_Ptr[2], dev_Ptr[i], i, index_d[i], int(idx/temp));
        }
    }
    
    // printf("idx=%d, dim[0]=%d, dim[1]=%d, dim[2]=%d \n", idx, dev_Ptr[0], dev_Ptr[1], dev_Ptr[2]);

    int data_index = 0;
    dev_Ptr[axis] = ((const int *)index)[idx];
    // printf("axis ----- dev_Ptr[%d]=%d \n",axis, dev_Ptr[axis]);
    for(int i = 0; i<index_nbDims; ++i)
    {
        int temp = 1;
        for(int j=i+1; j<index_nbDims; ++j)
        {
            temp = temp*data_d[j];
        }

        if (i == index_nbDims-1)
        {
            data_index = data_index + dev_Ptr[i];
        }else{
            data_index += dev_Ptr[i]*temp;
        }
        // printf("axis=%d , dev_Ptr[%d]=%d \n", idx, i, dev_Ptr[i]);
    }
    float out_value = ((const float *)data)[data_index];
    // printf("idx=%d, data_index=%d, dev_Ptr[axis]=%d, out_value=%f \n", idx, data_index, dev_Ptr[axis], out_value);
    output[idx] = out_value;
}

int GatherElements::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

    const Dims indexDims = inputDesc[1].dims;
    const Dims dataDims  = inputDesc[0].dims;
    int N = calculateNumSlices(indexDims);
    int blockSize = 512;
    int gridSize  = (N + blockSize - 1) / blockSize;

    if(axis_ <=0)
    {
        std::cout<<"GatherElements occur error,please check the calue of axis_, It must be greater than 1"<<std::endl;
        return 0;
    }

    int nbDims = indexDims.nbDims;

    int temp_Dims[outputDesc[0].dims.MAX_DIMS];
    std::memset(temp_Dims, 0, sizeof(int)*outputDesc[0].dims.MAX_DIMS);

    int *device_indexDims = nullptr;
    cudaMalloc(&device_indexDims, sizeof(int)*nbDims);

    int *device_dataDims = nullptr;
    cudaMalloc(&device_dataDims, sizeof(int)*nbDims);

    int *device_tempDims = nullptr;
    cudaMalloc(&device_tempDims, sizeof(int)*outputDesc[0].dims.MAX_DIMS);

    cudaMemcpy(device_tempDims, temp_Dims, sizeof(int)*outputDesc[0].dims.MAX_DIMS, cudaMemcpyHostToDevice);

    cudaMemcpy(device_indexDims, indexDims.d, sizeof(int)*nbDims, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dataDims, dataDims.d, sizeof(int)*nbDims, cudaMemcpyHostToDevice);
    

    GatherElementKernel<<<gridSize, blockSize, 0, stream>>>(inputs[dataTensorIdx], 
                                                            inputs[indexTensorIdx],
                                                            (float*)(outputs[0]),
                                                            device_indexDims,
                                                            nbDims,
                                                            device_tempDims,
                                                            device_dataDims,
                                                            N,
                                                            axis_);
    
    cudaStreamSynchronize(stream);

    cudaFree(device_indexDims);
    device_indexDims = nullptr;

    cudaFree(device_dataDims);
    device_dataDims = nullptr;

    cudaFree(device_tempDims);
    device_tempDims = nullptr;

    return 1;
}