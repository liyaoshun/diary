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
#include "kernel.h"
#include "scatterPlugin.h"
#include "cuda_fp16.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include "half.h"
#include <cstring>

#define CUBLAS_CHECK(condition)                                                                 \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t status = condition;                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                       \
            printf("%s %d CUBLAS FAIL %s\n", __FILE__, __LINE__, cublasGetErrorString(status)); \
        }                                                                                       \
    } while (0)

//this scatter kernel works on a 2d table writing rows 
//index is 1-D array
//updates is 2-D array
//output is 2-D array
//output[index[i]] = updates[i]
__global__ void scatterKernel(
    char* output,
    const char* updates,
    const int* indices,
    int pitch,
    int rowSize)
{
    int idx = indices[blockIdx.x];
    char* pDst = (char*)output + idx * pitch;
    const char* pSrc = updates + blockIdx.x * rowSize; 
    memcpy(pDst, pSrc, rowSize);
}

// Transform nd index to 1 - d index
__global__ void transformIdxKernel(
    int* output,
    const int* transformCoeff, // these are actually the output pitches of the respective dimensions
    const int* indices,
    int sliceRank)
{
    const int* idx = indices + sliceRank * blockIdx.x;
    int transformedIdx = 0;
    for (int i = 0; i < sliceRank; i++)
    {
        transformedIdx += idx[i] * transformCoeff[i];
    }
    output[blockIdx.x] = transformedIdx;
}


pluginStatus_t scatterNDInference( 
    cudaStream_t stream,
    int* transformCoeff,
    int nOutputDims,
    int sliceRank,        
    int nRows,
    int rowSize,
    int copySize,
    int sizeOfElementInBytes,         
    const void* index,
    const void* updates,
    const void* data,
    void* output,
    void* workspace)
{
    const int* _index = (const int*)(index);
    const char* _updates = (const char*)(updates);
    char* _output = (char*)(output);
    int* wo = (int*)(workspace);
    int* transformedIdx = wo + sizeof(int)*nOutputDims;
    int* deviceTransformCoeff = wo;    
    cudaMemcpy(workspace, transformCoeff, sizeof(int)*nOutputDims,cudaMemcpyHostToDevice );
    transformIdxKernel<<<nRows, 1, 0, stream>>>(transformedIdx, deviceTransformCoeff, _index, sliceRank);
    cudaMemcpy(output, data, copySize, cudaMemcpyDeviceToDevice);
    //assuming output pitch = rowSize i.e no padding
    scatterKernel<<<nRows, 1, 0, stream>>>(_output, _updates, transformedIdx, rowSize*4, rowSize*4);
    return STATUS_SUCCESS;
}


int32_t ScatterND::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{  
    int32_t transformCoeff[outputDesc[0].dims.MAX_DIMS];
    std::memset(transformCoeff, 0, sizeof(int32_t)*outputDesc[0].dims.MAX_DIMS);
    Dims IndexDims = inputDesc[indexTensorIdx].dims;
    
    Dims dataDims = inputDesc[dataTensorIdx].dims;

    int32_t indexRank = IndexDims.d[IndexDims.nbDims-1];
    ASSERT(indexRank <= dataDims.nbDims);

    int32_t nSlices = calculateNumSlices(IndexDims);
    int32_t rowSize = 1;
    int32_t copySize = calculateCopySize(dataDims);
    int32_t elementSizeInBytes = 1;
    switch (inputDesc->type)
    {
    case DataType::kFLOAT:
    case DataType::kINT32:
        elementSizeInBytes = 4;
        break;
    case DataType::kHALF:
        elementSizeInBytes = 2;
        break;
    case DataType::kINT8:
    case DataType::kBOOL:
        elementSizeInBytes = 1;
        break;
    }
    
    for (int i = indexRank; i < dataDims.nbDims; i++)
    {
        rowSize *= dataDims.d[i];
    }
    
    calculateTransformCoeff(dataDims, indexRank, transformCoeff);

    scatterNDInference(stream, transformCoeff, 
    dataDims.nbDims, 
    indexRank, 
    nSlices, 
    rowSize,  
    copySize, 
    elementSizeInBytes,  
    inputs[indexTensorIdx],
    inputs[updateTensorIdx],
    inputs[dataTensorIdx],
    outputs[0],
    workspace );    

    return 0;
}